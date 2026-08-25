"""Frozen-encoder classifier: DistilBERT embeddings + logistic head."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT / 'data' / 'classify'
REPORT = SPLIT_DIR / 'frozen_report.csv'
ENCODER = 'distilbert-base-uncased'
MAX_TOKENS = 256
BATCH_SIZE = 16
AI_LABEL = 1
SEED = 37


def embed(texts: list[str], tokenizer: object, model: object) -> np.ndarray:
    """Return mean-pooled encoder embeddings, masked over real tokens only.

    Args:
        texts (list[str]): Documents to encode.
        tokenizer (object): Matching tokenizer.
        model (object): Frozen encoder.

    Returns:
        np.ndarray: One embedding row per document.

    """
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch = tokenizer(
                texts[start : start + BATCH_SIZE],
                truncation=True,
                max_length=MAX_TOKENS,
                padding=True,
                return_tensors='pt',
            )
            hidden = model(**batch).last_hidden_state
            mask = batch['attention_mask'].unsqueeze(-1).float()
            out.append(((hidden * mask).sum(1) / mask.sum(1)).numpy())
            if start % (BATCH_SIZE * 20) == 0:
                logger.info('  encoded %d/%d', start, len(texts))
    return np.vstack(out)
