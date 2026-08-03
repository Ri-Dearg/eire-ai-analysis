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
LENGTH_BANDS = ((0, 150), (150, 300), (300, 10**9))
FALSE_POSITIVE_TARGETS = (0.01, 0.05)


OUTLETS = ('rte', 'irish_examiner', 'the_liberal', 'gript')
PRIMARY = ('binoculars', 'fastdetectgpt', 'perplexity')
ALL_DETECTORS = (*PRIMARY, 'radar')


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


def ensemble_calls(calls: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Combine per-detector binary calls.

    Args:
        calls (dict[str, np.ndarray]): Detector bool for the primary detectors.

    Returns:
        dict[str, np.ndarray]: majority (>=2), unanimous (all),
            any1 (>=1) boolean arrays.

    """
    stack = np.vstack([calls[detector].astype(int) for detector in PRIMARY])
    votes = stack.sum(axis=0)
    num = len(PRIMARY)
    return {
        'majority': votes >= num // 2 + 1,
        'unanimous': votes == num,
        'any1': votes >= 1,
    }


def _binary_calls(
    corpus: pd.DataFrame,
    scores: dict[str, pd.Series],
    threshold_df: pd.DataFrame,
    false_positive_rate: float,
) -> pd.DataFrame:
    """Return per-article binary results for each detector at one FPR target.

    Uses each article's **outlet** threshold. Adds ensemble columns.

    Args:
        corpus (pd.DataFrame): The collected db / dataframe.
        scores (dict[str, pd.Series]): Scoring of a data set.
        threshold_df (pd.DataFrame): The threshold df.
        false_positive_rate (float): False Positive Rate.

    Returns:
        pd.DataFrame: Dataframe with binary results for each detector.

    """
    calls: dict[str, np.ndarray] = {}
    threshold_fpr_df = threshold_df[threshold_df.fpr_target == false_positive_rate]
    for detector in ALL_DETECTORS:
        lut = threshold_fpr_df[threshold_fpr_df.detector == detector].set_index('outlet')[
            'threshold'
        ]
        article_thr = corpus['outlet'].map(lut).to_numpy(dtype=float)
        article_score = corpus['id'].map(scores[detector]).to_numpy(dtype=float)
        calls[detector] = np.isfinite(article_score) & (article_score >= article_thr)
    ensemble = ensemble_calls({detector: calls[detector] for detector in PRIMARY})
    output = corpus[['id', 'outlet', 'period', 'is_wire', 'word_count']].copy()
    for detector in ALL_DETECTORS:
        output[f'{detector}_call'] = calls[detector]
    for name, arr in ensemble.items():
        output[f'ensemble_{name}'] = arr
    return output


def _length_band(words: int) -> str:
    """Return the length-band label for a word count.

    Args:
        words (int): Number of words.

    Returns:
        str: Either string of length, or na.

    """
    for low, high in LENGTH_BANDS:
        if low <= words < high:
            return f'{low}-{high if high < 10**9 else "inf"}'
    return 'na'


def headline_table(calls: pd.DataFrame, call_col: str) -> pd.DataFrame:
    """Detected-rate per outlet x period: all / non-wire / by length band.

    Args:
        calls (pd.DataFrame): Per-article calls from :func:`_binary_calls`.
        call_col (str): Which call column to summarise.

    Returns:
        pd.DataFrame: Long-format detected rows.

    """
    df = calls.copy()
    df['word_count'] = df['word_count'].astype(int)
    df['band'] = df['word_count'].map(_length_band)
    df['wire'] = df['is_wire'].astype(str) == '1'
    records: list[dict] = []

    for (outlet, period), group in df.groupby(['outlet', 'period']):
        records.append(
            {
                'outlet': outlet,
                'period': period,
                'view': 'all',
                'n': len(group),
                'detected': float(group[call_col].mean()),
            }
        )

        num_wire = group[~group.wire]
        records.append(
            {
                'outlet': outlet,
                'period': period,
                'view': 'non_wire',
                'n': len(num_wire),
                'detected': float(num_wire[call_col].mean()) if len(num_wire) else np.nan,
            }
        )
        for band, group_band in group.groupby('band'):
            records.append(
                {
                    'outlet': outlet,
                    'period': period,
                    'view': f'len_{band}',
                    'n': len(group_band),
                    'detected': float(group_band[call_col].mean()),
                }
            )

    return pd.DataFrame(records)


def positive_rate(scores: np.ndarray, threshold: float) -> float:
    """Return the fraction of finite scores at or above threshold.

    Args:
        scores (np.ndarray): Scores as an array.
        threshold (float): Scoring threshold

    Returns:
        float: Scores beyond the threshold.

    """
    scoring = scores[np.isfinite(scores)]
    if scoring.size == 0:
        return float('nan')
    return float(np.mean(scoring >= threshold))


def ensemble_fpr_on_anchor(
    inputs: pd.DataFrame,
    scores: dict[str, pd.Series],
    threshold_df: pd.DataFrame,
    false_positive_rate: float,
    primary: tuple[str, ...] = PRIMARY,
    min_votes: int = 2,
) -> dict[str, float]:
    """Return each outlet's realised ensemble FPR on the human anchor.

    Args:
        inputs (pd.DataFrame): The score-input table (needs group/outlet).
        scores (dict[str, pd.Series]): {detector: id->score}.
        threshold_df (pd.DataFrame): Output of :func:`calibrate`.
        false_positive_rate (float): The per-detector FPR target the thresholds were set at.
        primary (tuple[str, ...]): Ensemble members.
        min_votes (int): Votes required to flag (2 = majority of 3, 1 = any-1).

    Returns:
        dict[str, float]: ``{outlet: realised ensemble FPR}``.

    """
    human = inputs[inputs.group == 'human']
    threshold_fpr_df = threshold_df[threshold_df.fpr_target == false_positive_rate]
    output: dict[str, float] = {}
    for outlet in OUTLETS:
        ids = human[human.outlet == outlet]['id']
        votes = np.zeros(len(ids), dtype=int)
        for detector in primary:
            threshold = threshold_fpr_df[
                (threshold_fpr_df.detector == detector)
                & (threshold_fpr_df.outlet == outlet)
            ]
            threshold_value = float(threshold['threshold'].iloc[0])
            scoring = ids.map(scores[detector]).to_numpy(dtype=float)
            votes += (np.isfinite(scoring) & (scoring >= threshold_value)).astype(int)
        output[outlet] = float((votes >= min_votes).mean())
    return output


def sanity_pre_fpr(
    headline: pd.DataFrame, ensemble_fpr: dict[str, float], false_positive_rate: float
) -> pd.DataFrame:
    """Compare each outlet's PRE-cell detected rate to the target FPR.

    Args:
        headline (pd.DataFrame): Headlines df.
        ensemble_fpr (dict[str, float]): The ensemble FPR per outlet.
        false_positive_rate (float): The target FPR for comparison.

    Returns:
        pd.DataFrame: False Positive sanity check table.

    """
    pre = headline[(headline.period == 'pre') & (headline.view == 'all')]
    return pd.DataFrame(
        [
            {
                'outlet': r.outlet,
                'pre_detected': r.detected,
                'ensemble_fpr_anchor': ensemble_fpr.get(r.outlet, float('nan')),
                'delta': r.detected - ensemble_fpr.get(r.outlet, float('nan')),
                'fpr_target': false_positive_rate,
                'delta_asposted': r.detected
                - false_positive_rate,  # superseded, kept for the report
            }
            for r in pre.itertuples()
        ]
    )


def threshold_at_fpr(human_scores: np.ndarray, fpr: float) -> float:
    """Return the score threshold giving false-positive rate on humans.

    Args:
        human_scores (np.ndarray): Detector scores on known-human text.
        fpr (float): Target false-positive rate in (0, 1).

    Returns:
        float: The score threshold.

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


def _load_scores(detector: str) -> pd.Series:
    """Return an Series from a detector checkpoint CSV.

    Args:
        detector (str): Detector name

    Returns:
        pd.Series: Detector scores as dataframe.

    """
    path = DETECTION_DIR / f'{detector}.csv'
    df = pd.read_csv(path)
    return df.set_index('id').iloc[:, 0].astype(float)


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
                record = {
                    'detector': detector,
                    'outlet': outlet,
                    'fpr_target': false_positive_rate,
                    'threshold': threshold,
                    'n_human': int(np.isfinite(human_scoring).sum()),
                    'fpr_realised': positive_rate(human_scoring, threshold),
                    'tpr': positive_rate(ai_scoring, threshold),
                }
                for model, group in ai.groupby('model'):
                    model_scoring = group['id'].map(scoring).to_numpy(dtype=float)
                    record[f'tpr_{model}'] = positive_rate(model_scoring, threshold)
                records.append(record)
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
    scores = {detector: _load_scores(detector) for detector in ALL_DETECTORS}

    threshold_df = calibrate(inputs, scores)
    threshold_df.to_csv(DETECTION_DIR / 'calibration_report.csv', index=False)

    corpus = inputs[inputs.group == 'corpus'].copy()

    for false_positive_rate in FALSE_POSITIVE_TARGETS:
        binary = _binary_calls(corpus, scores, threshold_df, false_positive_rate)
        tag = f'fpr{int(false_positive_rate * 100)}'
        binary.to_csv(DETECTION_DIR / f'detection_scores_{tag}.csv', index=False)
        headlines = headline_table(binary, 'ensemble_majority')
        headlines.to_csv(DETECTION_DIR / f'headline_{tag}.csv', index=False)
        anchor_fpr = ensemble_fpr_on_anchor(
            inputs,
            scores,
            threshold_df,
            false_positive_rate,
            primary=PRIMARY,
            min_votes=len(PRIMARY) // 2 + 1,
        )
        sanity_pre_fpr(headlines, anchor_fpr, false_positive_rate).to_csv(
            DETECTION_DIR / f'sanity_pre_fpr_{tag}.csv', index=False
        )
    logger.info(
        'wrote calibration_report, detection_scores, headline, sanity to %s',
        DETECTION_DIR,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
