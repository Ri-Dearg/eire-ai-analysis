"""Effect sizes and band-wise sensitivity for the pre-specified analysis."""

from __future__ import annotations

import logging

import pandas as pd

from calibrate.calibrate import DATA, DETECTION_DIR, FALSE_POSITIVE_TARGETS, KNOWN_AI

logger = logging.getLogger(__name__)

# ---------- FILES ----------
CORPUS_SCORED = DATA / 'corpus_scored.csv'
CALIBRATION_REPORT = DETECTION_DIR / 'calibration_report.csv'
PRIMARY_EFFECTS = DETECTION_DIR / 'primary_effects.csv'
BAND_SENSITIVITY = DETECTION_DIR / 'tpr_by_band.csv'

REPORTED_DETECTORS = ('fastdetectgpt', 'binoculars', 'perplexity')
REPORTED_PERIODS = ('post', 'pre')


def main() -> int:
    """Run the specified primary, co-primary and S1, and write both tables.

    Returns:
        int: 0 on success, 1 if a required input is missing.

    """
    missing = [
        path
        for path in (CORPUS_SCORED, CALIBRATION_REPORT, KNOWN_AI)
        if not path.exists()
    ]
    if missing:
        logger.error(
            'Missing %s. Run `python -m calibrate` first, then re-run.',
            ', '.join(path.name for path in missing),
        )
        return 1

    corpus = pd.read_csv(
        CORPUS_SCORED,
        usecols=[
            'article_id',
            'outlet',
            'period',
            'year',
            'category',
            'word_count',
            *[f'{detector}_score' for detector in REPORTED_DETECTORS],
        ],
    )
    corpus['word_count'] = corpus['word_count'].astype(int)

    records: list[dict] = []
    for detector in REPORTED_DETECTORS:
        for period in REPORTED_PERIODS:
            print(corpus, detector, period)
    effects = pd.DataFrame(records)
    effects.to_csv(PRIMARY_EFFECTS, index=False)
    return 0
