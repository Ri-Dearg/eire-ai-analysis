"""Fine-tune DeBERTa-v3-base to separate human from AI-generated news text.

Module heavily aided by AI development due to difficulty training on local architecture.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

logger = logging.getLogger(__name__)

# ---------- FILE SYSTEM ----------
ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT / 'data' / 'classify'
MODEL_DIR = ROOT / 'models' / 'deberta-v3-base-ai-detect'

# ---------- MODEL ----------
BASE_MODEL = 'microsoft/deberta-v3-base'
LABEL_NAMES = {0: 'human', 1: 'ai'}
AI_LABEL = 1

# ---------- HYPERPARAMETERS ----------
MAX_TOKENS = 512
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
SEED = 37

PROPOSAL_TARGET_F1 = 0.85
MIN_GRAD_LOGS = 3

# Fraction of training steps spent warming the learning rate up from zero.
WARMUP_FRACTION = 0.06
LOG_EVERY = 25

SGD_LEARNING_RATE = 1e-4
SGD_MOMENTUM = 0.9
WEIGHT_DECAY = 0.01

# Below this a logged loss is zero with floating-point noise, not a real value:
# the first failed run logged -7.5e-12 and 2.4e-21, which are zeros, not garbage.
NUMERICAL_ZERO = 1e-6
# A loss this small on a balanced binary task means perfect separation. Chance is
# ln(2) = 0.693, so anything near zero is a shortcut rather than learning.
CONVERGED_LOSS = 0.01
# Matches the floor in classify/dataset.py. Applied again here because harmonise()
# runs after that floor, so a row can pass it and then be stripped to nothing.
MIN_SPLIT_WORDS = 50

# Fraction of training steps spent warming the learning rate up from zero.
WARMUP_FRACTION = 0.06
LOG_EVERY = 25

SGD_LEARNING_RATE = float(os.environ.get('CLASSIFY_SGD_LR', '1e-4'))
SGD_MOMENTUM = 0.9
WEIGHT_DECAY = 0.01


def compute_metrics(predictions: object) -> dict[str, float]:
    """Return accuracy, precision, recall and F1 for the AI class.

    F1 is binary on the AI class rather than macro-averaged: the Proposal's 0.85 target
    is about detecting AI text, and a macro average lets strong performance on the human
    class disguise weak detection.

    Args:
        predictions (object): A transformers ``EvalPrediction``.

    Returns:
        dict[str, float]: Metric name to value.

    """
    logits, labels = predictions
    predicted = np.asarray(logits).argmax(axis=-1)
    precision, recall, f1, _unused = precision_recall_fscore_support(
        labels, predicted, average='binary', pos_label=AI_LABEL, zero_division=0
    )
    return {
        'accuracy': float(accuracy_score(labels, predicted)),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
    }


# SanityGuard entirely built by AI. The issue of running on Mac Architecture made
# runs quire long before I could see if results were being generated. THis aided in
# shortening the time before I could gauge the result.
class SanityGuard(TrainerCallback):
    """Abort the moment the run shows it is not learning.

    Attributes:
        started (float): Wall-clock time the run began, for the projection.
        grad_norms (list[float]): Every gradient norm logged so far.
        losses (list[float]): Every training loss logged so far.

    """

    def __init__(self) -> None:
        """Initialise the timer and the gradient-norm history."""
        self.started = time.time()
        self.grad_norms: list[float] = []
        self.losses: list[float] = []

    def on_log(
        self, _args: object, state: object, _control: object, **kwargs: object
    ) -> None:
        """Check the latest log line and project the finish time.

        Args:
            _args (object): Training arguments; unused, positional in the API.
            state (object): Trainer state, carrying step and max_steps.
            _control (object): Trainer control; unused, positional in the API.
            **kwargs (object): Carries ``logs``, the metrics for this step.

        Raises:
            RuntimeError: If the loss is meaningfully negative, or the gradient has
                been zero for several logs.

        """
        logs = kwargs.get('logs') or {}
        loss = logs.get('loss')
        if loss is not None:
            self.losses.append(float(loss))
            if loss < -NUMERICAL_ZERO:
                message = (
                    f'loss {loss:.3g} is negative — cross-entropy cannot be. The '
                    'forward pass is producing invalid values.'
                )
                raise RuntimeError(message)

        norm = logs.get('grad_norm')
        if norm is not None:
            self.grad_norms.append(float(norm))
        if len(self.grad_norms) < MIN_GRAD_LOGS or max(self.grad_norms) != 0:
            self._report_progress(state)
            return

        # A zero gradient has two completely different causes and pointing at the wrong
        # one costs hours. Cross-entropy at chance on a balanced binary task is ln(2) =
        # 0.693, so the loss beside the zero gradient says which it is.
        recent = self.losses[-MIN_GRAD_LOGS:] if self.losses else [float('nan')]
        # The latest loss, not the largest: an early step can still be mid-descent
        # while the run has already collapsed to zero.
        if abs(recent[-1]) < CONVERGED_LOSS:
            message = (
                f'gradient is zero because the loss is zero (recent: {recent}). The '
                'model has separated the classes perfectly within a few hundred '
                'examples, which for this task means a shortcut, not learning — some '
                'surface feature differs between the two classes. Compare the classes '
                'on whitespace, quote characters, punctuation and boilerplate before '
                'training again; do not blame the device.'
            )
        else:
            message = (
                f'gradient is zero while the loss sits at {recent[-1]:.4g}, near the '
                'ln(2) = 0.693 chance level. No signal is reaching the weights, so no '
                'further epoch can change anything. This one is the backend or the '
                'optimiser, not the data.'
            )
        if os.environ.get('CLASSIFY_ALLOW_ZERO_LOSS'):
            logger.warning(message)
            return
        raise RuntimeError(message)

    def _report_progress(self, state: object) -> None:
        """Log the step count and a projected finish time.

        Args:
            state (object): Trainer state, carrying step and max_steps.

        """
        step = getattr(state, 'global_step', 0)
        total = getattr(state, 'max_steps', 0)
        if step and total:
            elapsed = time.time() - self.started
            remaining = elapsed / step * (total - step)
            logger.info(
                'step %d/%d · %.1f s/step · about %.1f h remaining',
                step,
                total,
                elapsed / step,
                remaining / 3600,
            )


def _load_split(name: str, tokenizer: object) -> object:
    """Return one tokenised split.

    Args:
        name (str): Split name, matching ``<name>.csv`` in the split directory.
        tokenizer (object): The model's tokenizer.

    Returns:
        object: A tokenised ``datasets.Dataset``.

    Raises:
        FileNotFoundError: If the split has not been built.

    """
    path = SPLIT_DIR / f'{name}.csv'
    if not path.exists():
        message = f'{path.name} not found. Run `python -m classify.dataset` first.'
        raise FileNotFoundError(message)

    frame = pd.read_csv(path)[['text', 'label']]
    dataset = Dataset.from_pandas(frame.reset_index(drop=True), preserve_index=False)
    logger.info(
        '%-11s %5d examples (%d human / %d AI)',
        name,
        len(frame),
        int((frame['label'] == 0).sum()),
        int((frame['label'] == AI_LABEL).sum()),
    )
    return dataset.map(
        lambda batch: tokenizer(batch['text'], truncation=True, max_length=MAX_TOKENS),
        batched=True,
        remove_columns=['text'],
    )


def train() -> dict[str, float]:
    """Fine-tune the classifier on CPU and save the best checkpoint.

    Returns:
        dict[str, float]: Validation metrics for the selected checkpoint.

    Raises:
        FileNotFoundError: If the splits have not been built.

    """
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    datasets = {name: _load_split(name, tokenizer) for name in ('train', 'validation')}

    steps_per_epoch = max(1, len(datasets['train']) // BATCH_SIZE)
    total_steps = int(steps_per_epoch * EPOCHS)
    logger.info(
        '%s run: %d tokens · batch %d · %.4g epochs · %d steps',
        'MPS',
        MAX_TOKENS,
        BATCH_SIZE,
        EPOCHS,
        total_steps,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label=LABEL_NAMES,
        label2id={value: key for key, value in LABEL_NAMES.items()},
    )

    arguments = TrainingArguments(
        output_dir=str(MODEL_DIR / 'checkpoints'),
        # Metal has no pinned memory; leaving this on only prints a warning.
        dataloader_pin_memory=False,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        warmup_steps=int(total_steps * WARMUP_FRACTION),
        logging_steps=LOG_EVERY,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        seed=SEED,
        report_to=[],
        max_grad_norm=1.0,
    )

    optimiser = torch.optim.SGD(
        model.parameters(),
        lr=SGD_LEARNING_RATE,
        momentum=SGD_MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    trainer = Trainer(
        optimizers=(optimiser, None),
        model=model,
        args=arguments,
        train_dataset=datasets['train'],
        eval_dataset=datasets['validation'],
        data_collator=DataCollatorWithPadding(tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[SanityGuard()],
    )
    trainer.train(resume_from_checkpoint=bool(os.environ.get('CLASSIFY_RESUME')))
    metrics = trainer.evaluate()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    logger.info('saved to %s', MODEL_DIR)
    return metrics


def main() -> int:
    """Fine-tune the classifier.

    Returns:
        int: 0 on success, 1 if the splits are missing or the run did not learn.

    """
    try:
        metrics = train()
    except FileNotFoundError:
        logger.exception('missing input')
        return 1
    except RuntimeError:
        logger.exception('training aborted')
        return 1

    for key, value in sorted(metrics.items()):
        if key.startswith('eval_'):
            logger.info('  %-12s %.4f', key.removeprefix('eval_'), value)
    f1 = metrics.get('eval_f1', float('nan'))
    logger.info(
        'validation F1 = %.4f (Proposal target 0.85) — %s',
        f1,
        'met' if f1 >= PROPOSAL_TARGET_F1 else 'not met',
    )
    logger.info(
        'settings to record beside the result: %s · %d tokens · %.4g epochs · '
        'batch %d · lr %g · seed %d',
        BASE_MODEL,
        MAX_TOKENS,
        EPOCHS,
        BATCH_SIZE,
        LEARNING_RATE,
        SEED,
    )
    return 0


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )
    sys.exit(main())
