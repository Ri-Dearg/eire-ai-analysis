"""Per-outlet calibration, corpus scoring, and output tables for the detectors."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------- CONFIG ----------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
CALIBRATION_DIR = DATA / 'calibration'
DETECTION_DIR = DATA / 'detection'

CORPUS = DATA / 'corpus.csv'
HUMAN_PARSED = CALIBRATION_DIR / 'human_parsed.csv'
KNOWN_AI = CALIBRATION_DIR / 'known_ai.csv'
INPUTS = DETECTION_DIR / 'score_inputs.csv'

MIN_HUMAN_CHARS = 400

FALSE_POSITIVE_TARGETS = (0.01, 0.05)


OUTLETS = ('rte', 'irish_examiner', 'the_liberal', 'gript')
PRIMARY = ('binoculars', 'fastdetectgpt', 'radar')
ALL_DETECTORS = (*PRIMARY, 'perplexity')


def _human_usable_row(row: pd.Series, seen: set[str]) -> bool:  # noqa: PLR0911
    """Return True if a human-anchor row survives the corpus drop rules.

    The human anchor and the corpus pass identical filters
    from :func:`postprocess.curate.drop_reason`.

    Args:
        row (pd.Series): One row of human_parsed.csv.
        seen (set[str]): Raw body_sha1 hashes already kept; mutated in place.

    Returns:
        bool: True if the row is a usable anchor article.

    """
    if row.get('http_status') != '200':
        return False
    if int(row.get('body_len_raw') or 0) < MIN_HUMAN_CHARS:
        return False
    if not row.get('body_text', '').strip():
        return False
    outlet = row.get('outlet', '')
    if outlet == 'irish_examiner' and row.get('sub_excl') == '1':
        return False
    if outlet == 'gript' and row.get('gript_premium') == '1':
        return False
    if outlet == 'gript' and row.get('is_otd') == '1':
        return False
    body_hash = row.get('body_sha1', '')
    if body_hash:
        if body_hash in seen:
            return False
        seen.add(body_hash)
    return True


def human_anchor_df() -> pd.DataFrame:
    """Return the usable human-anchor rows in  order.

    Returns:
        pd.DataFrame: The kept rows of ``human_parsed.csv`` (original columns),
            in canonical order.

    """
    human_parsed = pd.read_csv(HUMAN_PARSED, dtype=str).fillna('')
    human_parsed = human_parsed.assign(
        _ord=human_parsed['published_date'].replace('', '9999'),
        _aid=pd.to_numeric(human_parsed['article_id'], errors='coerce'),
    ).sort_values(['_ord', '_aid'])
    seen: set[str] = set()
    mask = pd.Series(
        [_human_usable_row(row, seen) for _, row in human_parsed.iterrows()],
        index=human_parsed.index,
    )
    return human_parsed[mask].drop(columns=['_ord', '_aid'])


def threshold_at_fpr(human_scores: np.ndarray, fpr: float) -> float:
    """Return the score threshold giving false-positive rate on humans.

    Args:
        human_scores (np.ndarray): Detector scores on known-human text.
        fpr (float): Target false-positive rate in (0, 1).

    Returns:
        float: The score threshold (nan if no finite human scores).

    """
    score = human_scores[np.isfinite(human_scores)]
    if score.size == 0:
        return float('nan')
    return float(np.quantile(score, 1.0 - fpr, method='higher'))


def build_inputs() -> Path:
    """Assemble the id -> text table to score (human + AI + corpus).

    Returns:
        Path: The written ``score_inputs.csv``.

    """
    DETECTION_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    missing = [p for p in (HUMAN_PARSED, KNOWN_AI, CORPUS) if not p.exists()]
    if missing:
        message = f'build_inputs needs {missing}, please create and validate.'
        raise FileNotFoundError(message)

    for _, row in human_anchor_df().iterrows():
        body = row.get('body_text', '')
        rows.append(
            {
                'id': f'human:{row["outlet"]}:{row["article_id"]}',
                'group': 'human',
                'outlet': row['outlet'],
                'model': '',
                'period': 'pre',
                'is_wire': row.get('is_wire', '0'),
                'word_count': len(body.split()),
                'text': body,
            }
        )

    ai = pd.read_csv(KNOWN_AI, dtype=str).fillna('')
    for _, row in ai.iterrows():
        rows.append(
            {
                'id': f'ai:{row["id"]}',
                'group': 'ai',
                'outlet': '',
                'model': row['model'],
                'period': '',
                'is_wire': '0',
                'word_count': row['n_words'],
                'text': row['text'],
            }
        )

    corpus = pd.read_csv(CORPUS, dtype=str).fillna('')
    for _, row in corpus.iterrows():
        rows.append(
            {
                'id': f'corpus:{row["article_id"]}',
                'group': 'corpus',
                'outlet': row['outlet'],
                'model': '',
                'period': row['period'],
                'is_wire': row['is_wire'],
                'word_count': row['word_count'],
                'text': row['body_text'],
            }
        )

    output = pd.DataFrame(
        rows,
        columns=[
            'id',
            'group',
            'outlet',
            'model',
            'period',
            'is_wire',
            'word_count',
            'text',
        ],
    )

    output.to_csv(INPUTS, index=False)
    logger.info(
        'wrote %d score inputs (%d human, %d ai, %d corpus) -> %s',
        len(output),
        (output.group == 'human').sum(),
        (output.group == 'ai').sum(),
        (output.group == 'corpus').sum(),
        INPUTS,
    )
    return INPUTS


def calibrate(inputs: pd.DataFrame, scores: dict[str, pd.Series]) -> pd.DataFrame:
    """Compute per-detector, per-outlet thresholds, FPR and TPR.

    Args:
        inputs (pd.DataFrame): The score-input table (with`group/outlet).
        scores (dict[str, pd.Series]): {detector: id->score}.

    Returns:
        pd.DataFrame: One row per (detector, outlet, fpr_target) with the
            threshold, realised FPR, overall TPR, and per-model TPR columns.

    """
    human = inputs[inputs.group == 'human']
    ai = inputs[inputs.group == 'ai']
    records: list[dict] = []
    for detector in ALL_DETECTORS:
        scoring = scores[detector]
        ai_scoring = ai['id'].map(scoring).to_numpy(dtype=float)
        for outlet in OUTLETS:
            human_ids = human[human.outlet == outlet]['id']
            human_scoring = human_ids.map(scoring).to_numpy(dtype=float)
            for false_positive_rate in FALSE_POSITIVE_TARGETS:
                threshold = threshold_at_fpr(human_scoring, false_positive_rate)
    return pd.DataFrame(records)


def main() -> int:
    """Assemble inputs (if needed) and, once scores exist, emit all outputs."""
    if not INPUTS.exists():
        build_inputs()
    missing = [
        detector
        for detector in ALL_DETECTORS
        if not (DETECTION_DIR / f'{detector}.csv').exists()
    ]
    if missing:
        logger.error(
            'Detector score files missing: %s. Run `python -m detect.score` '
            'then re-run. Error: %s',
            ', '.join(missing),
            sys.stderr,
        )
        return 1
    inputs = pd.read_csv(INPUTS, dtype=str).fillna('')
    return 0
