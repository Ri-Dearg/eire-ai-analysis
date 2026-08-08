"""Effect sizes and band-wise sensitivity for the pre-specified analysis."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from calibrate.calibrate import (
    DATA,
    DETECTION_DIR,
    FALSE_POSITIVE_TARGETS,
    KNOWN_AI,
    LENGTH_BANDS,
)

logger = logging.getLogger(__name__)

# ---------- FILES ----------
CORPUS_SCORED = DATA / 'corpus_scored.csv'
CALIBRATION_REPORT = DETECTION_DIR / 'calibration_report.csv'
PRIMARY_EFFECTS = DETECTION_DIR / 'primary_effects.csv'
BAND_SENSITIVITY = DETECTION_DIR / 'tpr_by_band.csv'

PRIMARY_DETECTOR = 'fastdetectgpt'
REPORTED_DETECTORS = ('fastdetectgpt', 'binoculars', 'perplexity')
PRIMARY_PERIOD = 'post'
REPORTED_PERIODS = ('post', 'pre')

CONTROL_CATEGORY = 'legacy'
TREATMENT_CATEGORY = 'counter-consensus'

# ---------- SETTINGS ----------
SEED = 37
# Settings aided by AI
BOOTSTRAP_RESAMPLES = 10_000
MATERIALITY_THRESHOLD = 0.15


# ---------- LENGTH BANDS ----------
def band_label(word_count: int, bands: Sequence[tuple[int, int]] = LENGTH_BANDS) -> str:
    """Return the length-band label for a word count.

    Args:
        word_count (int): Article word count.
        bands (Sequence[tuple[int, int]]): ``[low, high)``.

    Returns:
        str: Label such as ``'0-150'`` or ``'300-inf'``; ``'na'`` if unmatched.

    """
    for low, high in bands:
        if low <= word_count < high:
            return f'{low}-{high if high < 10**9 else "inf"}'
    return 'na'


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


# ---------- CONTRAST ----------
def category_contrast(
    corpus: pd.DataFrame,
    detector: str,
    period: str,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> list[dict]:
    """Return the category contrast overall, per length band, per year and pooled.

    Args:
        corpus (pd.DataFrame): Scored corpus; needs ``category``, ``period``,
            ``year``, ``word_count`` and ``<detector>_score``.
        detector (str): Detector whose score column is compared.
        period (str): Period cell to restrict to, ``'post'`` or ``'pre'``.
        resamples (int): Bootstrap resamples for the interval.

    Returns:
        list[dict]: Records with delta, interval, p-value and group sizes.

    """
    cell = corpus[corpus['period'] == period]
    score_column = f'{detector}_score'
    records: list[dict] = []
    logger.info('%s / %s: overall contrast (n=%d)', detector, period, len(cell))
    records.append(contrast(cell, 'all', score_column, detector, period, resamples))
    print(records)

    for low, high in LENGTH_BANDS:
        band = band_label(low)
        in_band = cell[(cell['word_count'] >= low) & (cell['word_count'] < high)]
        logger.info('%s / %s: band %s (n=%d)', detector, period, band, len(in_band))
        records.append(
            contrast(in_band, f'band_{band}', score_column, detector, period, resamples)
        )


def contrast(
    frame: pd.DataFrame,
    scope: str,
    score_column: str,
    detector: str,
    period: str,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict:
    """Return one contrast record for a subset of the period cell.

    Args:
        frame (pd.DataFrame): Rows to contrast.
        scope (str): Label for the subset.
        score_column (str): Name of the score column to use.
        detector (str): The detector being analyzed.
        period (str): The period being analyzed.
        resamples (int): The number of bootstrap resamples.

    Returns:
        dict: One output row.

    """
    treatment = frame.loc[frame['category'] == TREATMENT_CATEGORY, score_column].to_numpy(
        dtype=float
    )
    control = frame.loc[frame['category'] == CONTROL_CATEGORY, score_column].to_numpy(
        dtype=float
    )
    delta, ci_low, ci_high = delta_ci(treatment, control, resamples=resamples)
    return {
        'detector': detector,
        'period': period,
        'scope': scope,
        'n_treatment': int(np.isfinite(treatment).sum()),
        'n_control': int(np.isfinite(control).sum()),
        'delta': delta,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'p_value': mannwhitney_p(treatment, control),
        'material': bool(abs(delta) >= MATERIALITY_THRESHOLD)
        if np.isfinite(delta)
        else False,
        'is_primary': detector == PRIMARY_DETECTOR
        and period == PRIMARY_PERIOD
        and scope == 'all',
    }


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
