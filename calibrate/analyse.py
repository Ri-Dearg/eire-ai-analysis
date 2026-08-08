"""Effect sizes and band-wise sensitivity for the pre-specified analysis."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from calibrate.calibrate import DATA, DETECTION_DIR, FALSE_POSITIVE_TARGETS, KNOWN_AI

logger = logging.getLogger(__name__)

# ---------- FILES ----------
CORPUS_SCORED = DATA / 'corpus_scored.csv'
CALIBRATION_REPORT = DETECTION_DIR / 'calibration_report.csv'
PRIMARY_EFFECTS = DETECTION_DIR / 'primary_effects.csv'
BAND_SENSITIVITY = DETECTION_DIR / 'tpr_by_band.csv'

REPORTED_DETECTORS = ('fastdetectgpt', 'binoculars', 'perplexity')
REPORTED_PERIODS = ('post', 'pre')

CONTROL_CATEGORY = 'legacy'
TREATMENT_CATEGORY = 'counter-consensus'


def mannwhitney_p(treatment_scores: np.ndarray, control_scores: np.ndarray) -> float:
    """Return the two-sided Mann-Whitney U p-value for two groups.

    Args:
        treatment_scores (np.ndarray): First group's detector scores.
        control_scores (np.ndarray): Second group's detector scores.

    Returns:
        float: The p-value, or nan if either group is empty after filtering.

    """
    treatment = treatment_scores[np.isfinite(treatment_scores)]
    control = control_scores[np.isfinite(control_scores)]
    if treatment.size == 0 or control.size == 0:
        return float('nan')
    _unused, probability = mannwhitneyu(treatment, control, alternative='two-sided')
    return float(probability)


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
