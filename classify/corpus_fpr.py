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


def load_pre_corpus() -> pd.DataFrame:
    """Return the PRE corpus cell, harmonised in memory.

    Returns:
        pd.DataFrame: One row per PRE article with a harmonised ``text`` column.

    Raises:
        FileNotFoundError: If the corpus is absent.

    """
    if not CORPUS.exists():
        message = f'{CORPUS} not found; the corpus is local-only.'
        raise FileNotFoundError(message)

    frame = pd.read_csv(CORPUS, usecols=list(CORPUS_COLUMNS))
    frame = frame[frame['period'] == PRE_PERIOD].copy()
    logger.info('PRE corpus rows: %d', len(frame))

    # Same transformation the classifier was trained under. In memory only.
    frame['text'] = frame['body_text'].fillna('').map(harmonise)
    frame['band'] = frame['word_count'].astype(int).map(band_label)
    frame = frame.drop(columns=['body_text'])
    return frame[frame['text'].str.strip().astype(bool)].copy()


def embed(texts: list[str]) -> np.ndarray:
    """Return mean-pooled frozen-encoder embeddings, masked over real tokens.

    Args:
        texts (list[str]): Documents to encode.

    Returns:
        np.ndarray: One embedding row per document.

    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ENCODER)
    model = AutoModel.from_pretrained(ENCODER).eval()  # CPU, inference only

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
            if start % (BATCH_SIZE * 100) == 0:
                logger.info('  encoded %d/%d', start, len(texts))
    return np.vstack(out)
