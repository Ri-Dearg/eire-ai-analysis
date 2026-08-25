"""Cross-validated per-generator recall with confidence intervals."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from statsmodels.stats.proportion import proportion_confint

logger = logging.getLogger(__name__)

# ---------- FILE SYSTEM ----------
ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT / 'data' / 'classify'
REPORT = SPLIT_DIR / 'crossval_report.csv'

# ---------- SETTINGS ----------
SEED = 37
FOLDS = 5
AI_LABEL = 1
HUMAN_LABEL = 0
CONFIDENCE = 0.05
MODELS = os.environ.get('CROSSVAL_MODEL', 'tfidf,frozen').split(',')

# Frozen-encoder settings, held identical to classify/frozen.py so the two are comparable.
ENCODER = 'distilbert-base-uncased'
MAX_TOKENS = 256
BATCH_SIZE = 16


def load_all() -> pd.DataFrame:
    """Return the three splits concatenated back into one frame.

    Cross-validation makes its own partition, so the stored split labels are discarded.

    Returns:
        pd.DataFrame: Every row from train, validation and test.

    Raises:
        FileNotFoundError: If the splits have not been built.

    """
    parts = []
    for name in ('train', 'validation', 'test'):
        path = SPLIT_DIR / f'{name}.csv'
        if not path.exists():
            message = f'{path} not found; run python -m classify.dataset first.'
            raise FileNotFoundError(message)
        parts.append(pd.read_csv(path))
    frame = pd.concat(parts, ignore_index=True)
    logger.info(
        'pooled %d rows (%d human / %d AI) across %d generators',
        len(frame),
        int((frame['label'] == HUMAN_LABEL).sum()),
        int((frame['label'] == AI_LABEL).sum()),
        frame.loc[frame['label'] == AI_LABEL, 'model'].nunique(),
    )
    return frame


def _groups(frame: pd.DataFrame) -> np.ndarray:
    """Return a grouping key that keeps one headline's generations together.

    Args:
        frame (pd.DataFrame): Pooled rows.

    Returns:
        np.ndarray: Group label per row.

    """
    prompt = frame['prompt_id'].fillna('').astype(str)
    # Anchor rows share no source, so each becomes its own group via its unique id.
    return np.where(prompt == '', 'anchor:' + frame['id'].astype(str), 'prompt:' + prompt)


def _embed(texts: list[str]) -> np.ndarray:
    """Return mean-pooled frozen-encoder embeddings.

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
            if start % (BATCH_SIZE * 50) == 0:
                logger.info('  encoded %d/%d', start, len(texts))
    return np.vstack(out)


def cross_validate(frame: pd.DataFrame, kind: str) -> pd.Series:
    """Return an out-of-fold prediction for every row.

    Args:
        frame (pd.DataFrame): Pooled rows.
        kind (str): ``'tfidf'`` or ``'frozen'``.

    Returns:
        pd.Series: Predicted label per row, aligned to ``frame``.

    """
    texts = frame['text'].astype(str).tolist()
    features = _embed(texts) if kind == 'frozen' else None

    predicted = np.zeros(len(frame), dtype=int)
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(frame, frame['label'], _groups(frame)), start=1
    ):
        if kind == 'frozen':
            model = LogisticRegression(max_iter=2000, random_state=SEED)
            model.fit(features[train_idx], frame['label'].to_numpy()[train_idx])
            predicted[test_idx] = model.predict(features[test_idx])
        else:
            model = make_pipeline(
                TfidfVectorizer(
                    max_features=50_000, ngram_range=(1, 2), sublinear_tf=True
                ),
                LogisticRegression(max_iter=2000, random_state=SEED),
            )
            model.fit([texts[i] for i in train_idx], frame['label'].to_numpy()[train_idx])
            predicted[test_idx] = model.predict([texts[i] for i in test_idx])
        logger.info(
            '  %s fold %d/%d done (%d held out)', kind, fold, FOLDS, len(test_idx)
        )
    return pd.Series(predicted, index=frame.index)
