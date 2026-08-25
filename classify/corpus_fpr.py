"""False-positive rate of the shipped classifiers on the pre-ChatGPT corpus cell."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from calibrate.calibrate import LENGTH_BANDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from classify.dataset import harmonise

logger = logging.getLogger(__name__)

# ---------- FILE SYSTEM ----------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
CORPUS = DATA / 'corpus.csv'
SPLIT_DIR = DATA / 'classify'
REPORT = SPLIT_DIR / 'corpus_fpr_report.csv'

# ---------- SETTINGS ----------
SEED = 37
AI_LABEL = 1
PRE_PERIOD = 'pre'
MODELS = os.environ.get('CORPUS_FPR_MODEL', 'tfidf').split(',')
# The boundary the shipped classifiers use, and the rate to calibrate against instead.
GLOBAL_BOUNDARY = 0.5
FPR_TARGET = 0.05
# Held identical to classify/distil.py so the two rates are comparable.
ENCODER = 'distilbert-base-uncased'
MAX_TOKENS = 256
BATCH_SIZE = 16
# Columns read from the corpus. body_text is the only large one; nothing else is needed.
CORPUS_COLUMNS = ('article_id', 'outlet', 'period', 'word_count', 'body_text')


def band_label(word_count: int, bands: tuple = LENGTH_BANDS) -> str:
    """Return the length-band label for a word count.

    Args:
        word_count (int): Words in the article.
        bands (tuple): ``(low, high)`` pairs, half-open.

    Returns:
        str: Band label, or ``'unbanded'`` if no band matches.

    """
    for low, high in bands:
        if low <= word_count < high:
            return f'{low}-{"inf" if high >= 10**9 else high}'
    return 'unbanded'


def load_training_frame() -> pd.DataFrame:
    """Return the three classifier splits concatenated.

    Returns:
        pd.DataFrame: Pooled training rows.

    Raises:
        FileNotFoundError: If the splits have not been built.

    """
    parts = []
    for name in ('train', 'validation', 'test'):
        path = SPLIT_DIR / f'{name}.csv'
        if not path.exists():
            message = f'{path} not found; run classify.dataset first.'
            raise FileNotFoundError(message)
        parts.append(pd.read_csv(path))
    frame = pd.concat(parts, ignore_index=True)
    logger.info(
        'training rows: %d (%d human / %d AI)',
        len(frame),
        int((frame['label'] != AI_LABEL).sum()),
        int((frame['label'] == AI_LABEL).sum()),
    )
    return frame
