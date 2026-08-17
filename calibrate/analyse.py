"""Effect sizes and band-wise sensitivity for the pre-specified analysis.

Produces Cliff's delta with a bootstrap interval and a
Mann-Whitney p-value, and a true-positive rate per length band, which the
co-primary needs because it compares detected rates band by band.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from calibrate.calibrate import (
    DATA,
    DETECTION_DIR,
    FALSE_POSITIVE_TARGETS,
    KNOWN_AI,
    LENGTH_BANDS,
    OUTLETS,
    PRIMARY,
    _load_scores,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

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

CATEGORY_LABELS = {'family': 'category', 'view': 'all', 'outlet': '(category)'}

# ---------- SETTINGS ----------
SEED = 37
# Settings aided by AI
BOOTSTRAP_RESAMPLES = 10_000
MATERIALITY_THRESHOLD = 0.15

# Below this many reference articles a band's true-positive rate is reported as
# not estimable rather than as a number.
MIN_BAND_ARTICLES = 30
TOP_BYLINES = 3


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


# ---------- BAND-WISE SENSITIVITY ----------
def tpr_by_band(
    reference: pd.DataFrame,
    scores: dict[str, pd.Series],
    thresholds: pd.DataFrame,
    detectors: Sequence[str],
    outlets: Sequence[str],
    false_positive_targets: Sequence[float],
    min_band_articles: int = MIN_BAND_ARTICLES,
) -> pd.DataFrame:
    """Return true-positive rate per detector, outlet, FPR target and length band.

    Args:
        reference (pd.DataFrame): Known-AI rows.
        scores (dict[str, pd.Series]): ``{detector: id -> score}``.
        thresholds (pd.DataFrame): Output of ``calibrate.calibrate``.
        detectors (Sequence[str]): Detector names.
        outlets (Sequence[str]): Outlet slugs.
        false_positive_targets (Sequence[float]): Target false-positive rates.
        min_band_articles (int): Floor below which a band is not estimable.

    Returns:
        pd.DataFrame: One row per (detector, outlet, fpr_target, band).

    """
    reference = reference.copy()
    reference['band'] = reference['word_count'].astype(int).map(band_label)
    records: list[dict] = []
    for detector in detectors:
        reference_scores = reference['id'].map(scores[detector]).to_numpy(dtype=float)
        for outlet in outlets:
            for false_positive_rate in false_positive_targets:
                match = thresholds[
                    (thresholds.detector == detector)
                    & (thresholds.outlet == outlet)
                    & (thresholds.fpr_target == false_positive_rate)
                ]
                threshold = float(match['threshold'].iloc[0])
                for band in reference['band'].unique():
                    in_band = (reference['band'] == band).to_numpy() & np.isfinite(
                        reference_scores
                    )
                    article_count = int(in_band.sum())
                    estimable = article_count >= min_band_articles
                    records.append(
                        {
                            'detector': detector,
                            'outlet': outlet,
                            'fpr_target': false_positive_rate,
                            'band': band,
                            'n_reference': article_count,
                            'estimable': estimable,
                            'tpr': float(np.mean(reference_scores[in_band] >= threshold))
                            if estimable
                            else float('nan'),
                        }
                    )
    return pd.DataFrame(records)


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


def byline_concentration(corpus: pd.DataFrame, top: int = TOP_BYLINES) -> pd.DataFrame:
    """Return byline concentration per outlet and period.

    Args:
        corpus (pd.DataFrame): Scored corpus; needs ``outlet``, ``period`` and
            ``author``.
        top (int): How many leading bylines to summarise.

    Returns:
        pd.DataFrame: One row per outlet x period.

    """
    records = []
    for (outlet, period), cell in corpus.groupby(['outlet', 'period'], sort=False):
        counts = cell.loc[cell['author'].notna(), 'author'].value_counts()
        records.append(
            {
                'outlet': outlet,
                'period': period,
                'n_articles': len(cell),
                'n_named': int(counts.sum()),
                'n_authors': int(counts.size),
                f'top{top}_articles': int(counts.head(top).sum()),
                f'top{top}_share': float(counts.head(top).sum() / len(cell))
                if len(cell)
                else float('nan'),
            }
        )
    return pd.DataFrame(records)


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

    for low, high in LENGTH_BANDS:
        band = band_label(low)
        in_band = cell[(cell['word_count'] >= low) & (cell['word_count'] < high)]
        logger.info('%s / %s: band %s (n=%d)', detector, period, band, len(in_band))
        records.append(
            contrast(in_band, f'band_{band}', score_column, detector, period, resamples)
        )

    # The POST cell opens in 2022: the split is ChatGPT's release, so December
    # 2022 is POST (382 articles).
    if period == PRIMARY_PERIOD:
        for year in sorted(cell['year'].unique()):
            in_year = cell[cell['year'] == year]
            logger.info('%s / %s: year %d (n=%d)', detector, period, year, len(in_year))
            records.append(
                contrast(
                    in_year, f'year_{year}', score_column, detector, period, resamples
                )
            )

    band_records = [row for row in records if row['scope'].startswith('band_')]
    weights = np.array(
        [row['n_treatment'] + row['n_control'] for row in band_records], dtype=float
    )
    deltas = np.array([row['delta'] for row in band_records], dtype=float)
    usable = np.isfinite(deltas) & (weights > 0)
    pooled = (
        float(np.average(deltas[usable], weights=weights[usable]))
        if usable.any()
        else float('nan')
    )
    records.append(
        {
            'detector': detector,
            'period': period,
            'scope': 'band_pooled',
            'n_treatment': sum(row['n_treatment'] for row in band_records),
            'n_control': sum(row['n_control'] for row in band_records),
            'delta': pooled,
            'ci_low': float('nan'),
            'ci_high': float('nan'),
            'p_value': float('nan'),
            'material': bool(abs(pooled) >= MATERIALITY_THRESHOLD)
            if np.isfinite(pooled)
            else False,
            'is_primary': False,
        }
    )
    return records


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


def non_wire_contrast(
    corpus: pd.DataFrame, detector: str, resamples: int = BOOTSTRAP_RESAMPLES
) -> list[dict]:
    """Return the category contrast on editorial (non-wire) articles only.

    Args:
        corpus (pd.DataFrame): Scored corpus; needs ``is_wire``, ``category``,
            ``period`` and ``<detector>_score``.
        detector (str): Detector whose score column is compared.
        resamples (int): Bootstrap resamples for the interval.

    Returns:
        list[dict]: One record per period.

    """
    score_column = f'{detector}_score'
    editorial = corpus[corpus['is_wire'].astype(int) == 0]
    records = []

    for period in REPORTED_PERIODS:
        cell = editorial[editorial['period'] == period]
        slices = [('all', cell)]
        if period == PRIMARY_PERIOD:
            slices += [
                (f'year_{year}', cell[cell['year'] == year])
                for year in sorted(cell['year'].unique())
            ]
        for scope, slice_rows in slices:
            logger.info(
                '%s: non_wire / %s %s (n=%d)', detector, period, scope, len(slice_rows)
            )
            row = contrast(slice_rows, scope, score_column, detector, period, resamples)
            row.update(
                {
                    'family': 'sensitivity',
                    'view': 'non_wire',
                    'outlet': '(category)',
                    'is_primary': False,
                }
            )
            records.append(row)

    return records


def outlet_contrast(
    corpus: pd.DataFrame, detector: str, resamples: int = BOOTSTRAP_RESAMPLES
) -> list[dict]:
    """Return each outlet against its own pre-ChatGPT baseline, year by year.

    Args:
        corpus (pd.DataFrame): Scored corpus.
        detector (str): Detector whose score column is compared.
        resamples (int): Bootstrap resamples for the interval.

    Returns:
        list[dict]: One record per outlet per POST slice.

    """
    score_column = f'{detector}_score'
    records: list[dict] = []

    for outlet in OUTLETS:
        rows = corpus[corpus['outlet'] == outlet]
        post = rows[rows['period'] == PRIMARY_PERIOD]
        pre = rows[rows['period'] == 'pre']

        slices = [('post_all', post, pre)]
        slices += [
            (f'year_{year}', post[post['year'] == year], pre)
            for year in sorted(post['year'].unique())
        ]

        for low, high in LENGTH_BANDS:
            band = band_label(low)
            post_band = post[(post['word_count'] >= low) & (post['word_count'] < high)]
            pre_band = pre[(pre['word_count'] >= low) & (pre['word_count'] < high)]
            slices.append((f'band_{band}', post_band, pre_band))
            slices += [
                (
                    f'band_{band}_year_{year}',
                    post_band[post_band['year'] == year],
                    pre_band,
                )
                for year in sorted(post_band['year'].unique())
            ]

        for scope, slice_rows, baseline_rows in slices:
            logger.info('%s: %s %s (n=%d)', detector, outlet, scope, len(slice_rows))
            own_pre = baseline_rows[score_column].to_numpy(float)
            slice_scores = slice_rows[score_column].to_numpy(float)
            delta, ci_low, ci_high = delta_ci(slice_scores, own_pre, resamples=resamples)
            records.append(
                {
                    'family': 'outlet',
                    'view': 'own_pre',
                    'outlet': outlet,
                    'detector': detector,
                    'period': PRIMARY_PERIOD,
                    'scope': scope,
                    'n_treatment': int(np.isfinite(slice_scores).sum()),
                    'n_control': int(np.isfinite(own_pre).sum()),
                    'delta': delta,
                    'ci_low': ci_low,
                    'ci_high': ci_high,
                    'p_value': mannwhitney_p(slice_scores, own_pre),
                    'material': bool(abs(delta) >= MATERIALITY_THRESHOLD)
                    if np.isfinite(delta)
                    else False,
                    'is_primary': False,
                }
            )
    return records


def pre_year_contrast(
    corpus: pd.DataFrame, detector: str, resamples: int = BOOTSTRAP_RESAMPLES
) -> list[dict]:
    """Return the category contrast within each PRE year.

    Calls :func:`contrast` unchanged, so a PRE year delta and the primary are the
    same computation on different rows.

    Args:
        corpus (pd.DataFrame): Scored corpus.
        detector (str): Detector whose score column is compared.
        resamples (int): Bootstrap resamples for the interval.

    Returns:
        list[dict]: One record per PRE year.

    """
    score_column = f'{detector}_score'
    cell = corpus[corpus['period'] == 'pre']
    records = []
    for year in sorted(cell['year'].unique()):
        in_year = cell[cell['year'] == year]
        logger.info('%s: PRE year %d (n=%d)', detector, year, len(in_year))
        row = contrast(in_year, f'year_{year}', score_column, detector, 'pre', resamples)
        row.update(
            {
                'family': 'pre_year',
                'view': 'all',
                'outlet': '(category)',
                'is_primary': False,
            }
        )
        records.append(row)
    return records


def main() -> int:
    """Run the statistic tables.

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
            'is_wire',
            'author',
            *[f'{detector}_score' for detector in REPORTED_DETECTORS],
        ],
    )
    corpus['word_count'] = corpus['word_count'].astype(int)

    records: list[dict] = []
    for detector in REPORTED_DETECTORS:
        for period in REPORTED_PERIODS:
            for row in category_contrast(corpus, detector, period):
                row.update(CATEGORY_LABELS)
                records.append(row)
        records.extend(pre_year_contrast(corpus, detector))
        records.extend(outlet_contrast(corpus, detector))
        records.extend(non_wire_contrast(corpus, detector))

    effects = pd.DataFrame(records)
    effects.to_csv(PRIMARY_EFFECTS, index=False)

    reference = pd.read_csv(KNOWN_AI, usecols=['id', 'n_words'])
    reference = reference.rename(columns={'n_words': 'word_count'})
    reference['id'] = 'ai:' + reference['id'].astype(str)
    scores = {detector: _load_scores(detector) for detector in PRIMARY}
    thresholds = pd.read_csv(CALIBRATION_REPORT)
    tpr_by_band(
        reference, scores, thresholds, PRIMARY, OUTLETS, FALSE_POSITIVE_TARGETS
    ).to_csv(BAND_SENSITIVITY, index=False)

    primary = effects[effects.is_primary].iloc[0]
    logger.info(
        'primary (%s, %s, %s vs %s): delta=%+.3f [%+.3f, %+.3f] p=%.4g material=%s',
        PRIMARY_DETECTOR,
        PRIMARY_PERIOD,
        TREATMENT_CATEGORY,
        CONTROL_CATEGORY,
        primary.delta,
        primary.ci_low,
        primary.ci_high,
        primary.p_value,
        primary.material,
    )
    logger.info(
        'byline concentration:\n%s', byline_concentration(corpus).to_string(index=False)
    )
    logger.info('wrote %s and %s', PRIMARY_EFFECTS.name, BAND_SENSITIVITY.name)
    return 0


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )
    sys.exit(main())
