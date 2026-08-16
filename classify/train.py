"""Fine-tune DeBERTa-v3-base to separate human from AI-generated news text."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
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
    )
    metrics = trainer.evaluate()
    return metrics


def main() -> int:
    """Fine-tune the classifier.

    Returns:
        int: 0 on success, 1 if the splits are missing.

    """
    try:
        return 1
    except FileNotFoundError:
        logger.exception('missing input')
        return 1
    return 0


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )
    sys.exit(main())
