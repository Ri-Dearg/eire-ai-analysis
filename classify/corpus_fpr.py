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


def probabilities(train: pd.DataFrame, corpus: pd.DataFrame, kind: str) -> np.ndarray:
    """Return P(AI) for every PRE-corpus row.

    Args:
        train (pd.DataFrame): Pooled training rows.
        corpus (pd.DataFrame): Harmonised PRE corpus rows.
        kind (str): ``'tfidf'`` or ``'frozen'``.

    Returns:
        np.ndarray: Probability of the AI class, one per corpus row.

    """
    train_text = train['text'].astype(str).tolist()
    corpus_text = corpus['text'].astype(str).tolist()

    if kind == 'frozen':
        logger.info('encoding %d training rows', len(train_text))
        x_train = embed(train_text)
        logger.info('encoding %d corpus rows -- this is the slow part', len(corpus_text))
        x_corpus = embed(corpus_text)
        model = LogisticRegression(max_iter=2000, random_state=SEED)
        model.fit(x_train, train['label'])
    else:
        model = make_pipeline(
            TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True),
            LogisticRegression(max_iter=2000, random_state=SEED),
        ).fit(train_text, train['label'])
        x_corpus = corpus_text

    ai_column = list(model.classes_).index(AI_LABEL)
    return model.predict_proba(x_corpus)[:, ai_column]


def outlet_thresholds(
    corpus: pd.DataFrame, proba: np.ndarray, kind: str, target: float = FPR_TARGET
) -> list[dict]:
    """Return the per-outlet decision threshold giving a common false-positive rate.

    Args:
        corpus (pd.DataFrame): Harmonised PRE corpus rows, with ``outlet``.
        proba (np.ndarray): P(AI) per row, aligned to ``corpus``.
        kind (str): Model label for the report.
        target (float): Common false-positive rate to calibrate to.

    Returns:
        list[dict]: One record per outlet.

    """
    frame = corpus.assign(p_ai=proba)
    records = []
    for outlet, cell in frame.groupby('outlet', sort=True):
        threshold = float(np.quantile(cell['p_ai'], 1.0 - target, method='higher'))
        records.append(
            {
                'model': kind,
                'scope': 'outlet_threshold',
                'outlet': outlet,
                'band': '(all)',
                'n': len(cell),
                'false_positive_rate': float((cell['p_ai'] >= threshold).mean()),
                'threshold': threshold,
                'fpr_at_global_boundary': float((cell['p_ai'] >= GLOBAL_BOUNDARY).mean()),
            }
        )
    return records


def summarise(corpus: pd.DataFrame, predicted: np.ndarray, kind: str) -> list[dict]:
    """Return false-positive rates overall, per outlet, per band and per outlet x band.

    Args:
        corpus (pd.DataFrame): Harmonised PRE corpus rows.
        predicted (np.ndarray): Predicted labels, aligned to ``corpus``.
        kind (str): Model label for the report.

    Returns:
        list[dict]: One record per scope.

    """
    frame = corpus.assign(flagged=(predicted == AI_LABEL))
    rows = [
        {
            'model': kind,
            'scope': 'overall',
            'outlet': '(all)',
            'band': '(all)',
            'n': len(frame),
            'false_positive_rate': float(frame['flagged'].mean()),
        }
    ]
    for outlet, cell in frame.groupby('outlet', sort=True):
        rows.append(
            {
                'model': kind,
                'scope': 'outlet',
                'outlet': outlet,
                'band': '(all)',
                'n': len(cell),
                'false_positive_rate': float(cell['flagged'].mean()),
            }
        )
    for band, cell in frame.groupby('band', sort=True):
        rows.append(
            {
                'model': kind,
                'scope': 'band',
                'outlet': '(all)',
                'band': band,
                'n': len(cell),
                'false_positive_rate': float(cell['flagged'].mean()),
            }
        )
    for (outlet, band), cell in frame.groupby(['outlet', 'band'], sort=True):
        rows.append(
            {
                'model': kind,
                'scope': 'outlet_band',
                'outlet': outlet,
                'band': band,
                'n': len(cell),
                'false_positive_rate': float(cell['flagged'].mean()),
            }
        )
    return rows


def main() -> int:
    """Score both classifiers on the PRE corpus cell and write the report.

    Returns:
        int: 0 on success, 1 if inputs are missing.

    """
    try:
        train = load_training_frame()
        corpus = load_pre_corpus()
    except FileNotFoundError:
        logger.exception('missing input')
        return 1

    records: list[dict] = []
    for kind in MODELS:
        kind = kind.strip()
        if kind not in {'tfidf', 'frozen'}:
            logger.warning('unknown model %r, skipping', kind)
            continue
        logger.info('scoring PRE corpus with %s', kind)
        proba = probabilities(train, corpus, kind)
        records.extend(summarise(corpus, (proba >= GLOBAL_BOUNDARY).astype(int), kind))
        records.extend(outlet_thresholds(corpus, proba, kind))

    if not records:
        logger.error('no models run; set CORPUS_FPR_MODEL')
        return 1

    report = pd.DataFrame(records)
    report.to_csv(REPORT, index=False)

    for kind in report['model'].unique():
        subset = report[report['model'] == kind]
        overall = subset[subset['scope'] == 'overall']['false_positive_rate'].iloc[0]
        logger.info('%s: PRE-corpus false-positive rate %.4f', kind, overall)
        for row in subset[subset['scope'] == 'outlet'].itertuples():
            logger.info(
                '    %-16s %.4f  (n=%d)', row.outlet, row.false_positive_rate, row.n
            )
        for row in subset[subset['scope'] == 'band'].itertuples():
            logger.info(
                '    band %-11s %.4f  (n=%d)', row.band, row.false_positive_rate, row.n
            )
        thresholds = subset[subset['scope'] == 'outlet_threshold']
        if not thresholds.empty:
            logger.info(
                '  per-outlet thresholds at a common %.0f%% FPR:', FPR_TARGET * 100
            )
            for row in thresholds.itertuples():
                logger.info(
                    '    %-16s threshold %.3f  (global 0.5 gives %.4f)',
                    row.outlet,
                    row.threshold,
                    row.fpr_at_global_boundary,
                )
    logger.info('wrote %s', REPORT.name)
    return 0


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )
    sys.exit(main())
