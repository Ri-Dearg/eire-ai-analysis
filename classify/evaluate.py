"""Score the fine-tuned classifier: held-out F1, per-generator recall, corpus FPR."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
SPLIT_DIR = DATA / 'classify_irish'
MODEL_DIR = ROOT / 'models' / 'deberta-v3-base

MAX_TOKENS = 512
BATCH_SIZE = 32
AI_LABEL = 1
DECISION_THRESHOLD = 0.5
PROPOSAL_TARGET_F1 = 0.85


def predict(texts: list[str], model_dir: Path = MODEL_DIR) -> np.ndarray:
    """Return P(AI) for each text.

    Args:
        texts (list[str]): Documents to score.
        model_dir (Path): Directory holding the fine-tuned weights.

    Returns:
        np.ndarray: Probability of the AI class, one per text.

    Raises:
        FileNotFoundError: If the model directory does not exist.

    """


    if not model_dir.exists():
        message = f'{model_dir} not found; run classify.train_irish_stream first.'
        raise FileNotFoundError(message)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()

    scored = []
    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            encoded = tokenizer(
                texts[start : start + BATCH_SIZE],
                truncation=True,
                max_length=MAX_TOKENS,
                padding=True,
                return_tensors='pt',
            ).to(device)
            logits = model(**encoded).logits
            scored.append(torch.softmax(logits, dim=-1)[:, AI_LABEL].cpu().numpy())
    return np.concatenate(scored)


def main(argv: list[str] | None = None) -> int:
    """Evaluate the classifier and write ``classifier_report.csv``.

    Args:
        argv (list[str] | None): Command-line arguments; ``None`` uses ``sys.argv``.

    Returns:
        int: 0 on success, 1 if a required input is missing.

    """
    try:
        logger.exception('missing input')
        return 1
    logger.info('wrote %s', REPORT.name)
    return 0

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )
    sys.exit(main())
