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

# ---------- SETTINGS ----------
SEED = 37
BOOTSTRAP_RESAMPLES = 10_000


# ---------- EFFECT SIZE ----------
def cliffs_delta(treatment_scores: np.ndarray, control_scores: np.ndarray) -> float:
    """Return Cliff's delta for treatment against control.

    Args:
        treatment_scores (np.ndarray): First group's detector scores.
        control_scores (np.ndarray): Second group's detector scores.

    Returns:
        float: Cliff's delta, or nan if either group is empty after filtering.

    """
    treatment = treatment_scores[np.isfinite(treatment_scores)]
    control = control_scores[np.isfinite(control_scores)]
    if treatment.size == 0 or control.size == 0:
        return float('nan')
    statistic, _unused = mannwhitneyu(treatment, control, alternative='two-sided')
    return float(2.0 * statistic / (treatment.size * control.size) - 1.0)


# Calculation Logic aided by AI
def delta_ci(
    treatment_scores: np.ndarray,
    control_scores: np.ndarray,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = SEED,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return Cliff's delta with a percentile bootstrap confidence interval.

    Args:
        treatment_scores (np.ndarray): First group's detector scores.
        control_scores (np.ndarray): Second group's detector scores.
        resamples (int): Bootstrap resamples; the specification fixes 10,000.
        seed (int): RNG seed for a reproducible interval.
        alpha (float): Two-sided error rate; 0.05 gives a 95% interval.

    Returns:
        tuple[float, float, float]: ``(delta, ci_low, ci_high)``.

    """
    treatment = treatment_scores[np.isfinite(treatment_scores)]
    control = control_scores[np.isfinite(control_scores)]
    point_estimate = cliffs_delta(treatment, control)
    if not np.isfinite(point_estimate):
        return point_estimate, float('nan'), float('nan')
    seeded_rng = np.random.default_rng(seed)
    resampled_deltas = np.empty(resamples, dtype=float)
    for index in range(resamples):
        treatment_sample = treatment[
            seeded_rng.integers(0, treatment.size, treatment.size)
        ]
        control_sample = control[seeded_rng.integers(0, control.size, control.size)]
        resampled_deltas[index] = cliffs_delta(treatment_sample, control_sample)
    ci_low, ci_high = np.quantile(resampled_deltas, [alpha / 2, 1 - alpha / 2])
    return point_estimate, float(ci_low), float(ci_high)


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
