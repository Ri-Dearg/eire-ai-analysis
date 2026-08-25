# ruff: noqa: RUF001  (en dashes and minus signs are intentional in labels)
"""Reporting pages: effect sizes, calibration, prevalence, generation, classifier."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd
import streamlit as st

from dashboard import theming
from dashboard.filters import VIEW_LABEL, pick_view

if TYPE_CHECKING:
    from dashboard.filters import Filters

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
DETECTION = DATA / 'detection'
GENERATION = DATA / 'generation'
CLASSIFY = DATA / 'classify'

PRIMARY_EFFECTS = DETECTION / 'primary_effects.csv'
CALIBRATION_REPORT = DETECTION / 'calibration_report.csv'
BAND_TPR = DETECTION / 'tpr_by_band.csv'
GENERATED_IRISH = GENERATION / 'generated_irish_ai.csv'
GENERATED_FRONTIER = GENERATION / 'generated_frontier_ai.csv'

CLASSIFIER_REPORTS = (
    CLASSIFY / 'crossval_report.csv',
    CLASSIFY / 'frozen_report.csv',
    CLASSIFY / 'baseline_report.csv',
)

# Tables written once per false-positive target. The sidebar control picks the suffix.
HEADLINE_STEM = 'headline_adjusted'
SANITY_STEM = 'sanity_pre_fpr'
RULES_STEM = 'rule_comparison'

MATERIALITY: Final = theming.MATERIALITY
# Cell value above which heatmap annotations need dark ink to stay legible.
HEATMAP_INK_FLIP: Final = 0.6

# The Irish-register generation ladder: five open-weight models plus two frontier
LADDER_TARGET_PER_MODEL: Final = 600
LADDER_MODELS: Final = (
    'gpt2-large',
    'gptneo-1.3b',
    'llama2-7b',
    'mistral-7b',
    'qwen2.5-7b',
    'gpt-4.1',
    'claude-opus-4-6',
)

LADDER_COLUMNS: Final = ('id', 'outlet', 'model', 'era', 'period', 'word_count')


# ---------- LOADING ----------
@st.cache_data(show_spinner=False)
def read(path: Path, columns: tuple[str, ...] | None = None) -> pd.DataFrame | None:
    """Return a CSV as a frame, or ``None`` if it is not on disk.

    Args:
        path (Path): File to read.
        columns (tuple[str, ...] | None): Column subset to read, or ``None`` for all.
            Columns absent from the file are skipped rather than raising.

    Returns:
        pd.DataFrame | None: The table, or ``None`` when absent.

    """
    if not path.exists():
        return None
    if columns is None:
        return pd.read_csv(path)
    header = pd.read_csv(path, nrows=0).columns
    present = [name for name in columns if name in header]
    return pd.read_csv(path, usecols=present)


def read_for_fpr(stem: str, filters: Filters) -> pd.DataFrame | None:
    """Return the variant of a per-target table matching the selected FPR.

    Args:
        stem (str): Filename stem, e.g. ``headline_adjusted``.
        filters (Filters): The active filter state.

    Returns:
        pd.DataFrame | None: The table, or ``None`` when absent.

    """
    return read(DETECTION / f'{stem}_{filters.suffix}.csv')


def awaiting(title: str, produced_by: str, shows: str) -> None:
    """Render a placeholder in place of a panel whose data has not arrived.

    Args:
        title (str): What the panel will show.
        produced_by (str): The command or step that produces the data.
        shows (str): One line on why the panel is worth having.

    """
    with st.container(border=True):
        st.markdown(f'#### {title}')
        st.caption(shows)
        st.code(produced_by, language='bash')
        st.caption('_Awaiting data — this panel fills in automatically once it exists._')


def _empty(message: str) -> None:
    """Explain that the current filter selection leaves a panel with nothing to draw.

    Args:
        message (str): What would need to change.

    """
    st.info(
        f'Nothing to show under the current filters — {message}',
        icon=':material/filter_alt_off:',
    )


# ---------- EFFECT SIZES ----------
def panel_primary(effects: pd.DataFrame) -> None:
    """Show the pre-specified primary as headline metrics.

    Args:
        effects (pd.DataFrame): The effects table, already filtered.

    """
    primary = effects[effects['is_primary']]
    if primary.empty:
        _empty(
            'the primary is Fast-DetectGPT over the post period, so keep that '
            'detector selected and the period on **both** or **post**.'
        )
        return
    row = primary.iloc[0]
    columns = st.columns(4)
    columns[0].metric("Cliff's δ", f'{row.delta:+.3f}')
    columns[1].metric('95% CI', f'{row.ci_low:+.3f} … {row.ci_high:+.3f}')
    columns[2].metric('p (Mann-Whitney)', f'{row.p_value:.4f}')
    columns[3].metric('Material (|δ| ≥ 0.15)', 'yes' if row.material else 'no')
    st.caption(
        f'Fast-DetectGPT · post-ChatGPT · counter-consensus (n = {row.n_treatment:,}) '
        f'against legacy (n = {row.n_control:,}) · no length floor. '
        'Positive δ means counter-consensus outlets score more AI-like. '
        'This row is fixed by the specification and does not move with the sidebar.'
    )


def panel_forest(effects: pd.DataFrame) -> None:
    """Plot every category-level contrast as an interval chart.

    Args:
        effects (pd.DataFrame): The effects table, already filtered.

    """
    rows = effects[
        (effects['family'] == 'category')
        & (effects['scope'].isin(['all', 'band_pooled']))
    ].copy()
    if rows.empty:
        _empty('no category-level contrast survives the detector and period selection.')
        return
    rows['label'] = (
        rows['detector'].map(theming.DETECTOR_LABEL)
        + ' · '
        + rows['period'].str.upper()
        + rows['scope'].map({'band_pooled': ' · pooled bands', 'all': ''}).fillna('')
    )
    rows = rows.sort_values(['detector', 'period', 'scope'])
    figure, axis = theming.figure(height=0.42 * len(rows) + 1.2)
    positions = np.arange(len(rows))
    errors = np.vstack(
        [
            (rows['delta'] - rows['ci_low'].fillna(rows['delta'])).to_numpy(),
            (rows['ci_high'].fillna(rows['delta']) - rows['delta']).to_numpy(),
        ]
    )
    axis.errorbar(
        rows['delta'],
        positions,
        xerr=np.abs(errors),
        fmt='o',
        capsize=3,
        color=theming.ink()['accent'],
    )
    theming.materiality_guides(axis)
    axis.set_yticks(positions)
    axis.set_yticklabels(rows['label'], fontsize=9)
    axis.invert_yaxis()
    axis.set_xlabel("Cliff's δ  (dotted: materiality threshold)")
    theming.show(figure)


def panel_bands(effects: pd.DataFrame, filters: Filters) -> None:
    """Plot the co-primary: the same contrast within each length band.

    Args:
        effects (pd.DataFrame): The effects table, already filtered.
        filters (Filters): The active filter state.

    """
    rows = effects[
        (effects['family'] == 'category')
        & (effects['period'] == 'post')
        & effects['scope'].str.startswith('band_')
        & (effects['scope'] != 'band_pooled')
    ]
    if rows.empty:
        _empty('the co-primary is written for the post period only.')
        return
    detectors = [d for d in filters.ordered_detectors() if d in set(rows['detector'])]
    if not detectors:
        _empty('none of the selected detectors has band rows.')
        return
    figure, axis = theming.figure(height=4.2)
    bands = sorted(
        rows['scope'].unique(), key=lambda name: int(name.split('_')[1].split('-')[0])
    )
    colours = theming.detector_colours()
    width = 0.8 / len(detectors)
    for index, detector in enumerate(detectors):
        subset = rows[rows['detector'] == detector].set_index('scope').reindex(bands)
        axis.bar(
            np.arange(len(bands)) + index * width,
            subset['delta'],
            width=width,
            label=theming.DETECTOR_LABEL[detector],
            color=colours[detector],
        )
    theming.materiality_guides(axis, vertical=False)
    axis.set_xticks(np.arange(len(bands)) + width * (len(detectors) - 1) / 2)
    axis.set_xticklabels([band.removeprefix('band_') for band in bands])
    axis.set_xlabel('length band (words)')
    axis.set_ylabel("Cliff's δ")
    axis.legend(fontsize=9)
    theming.show(figure)
    st.caption(
        'The pooled band estimate can disagree in sign with the unstratified primary. '
        'That is composition, not contradiction — the length mix itself moved across '
        'the period boundary.'
    )


def panel_year_trend(effects: pd.DataFrame, filters: Filters) -> None:
    """Plot the category contrast year by year, across the period boundary.

    Args:
        effects (pd.DataFrame): The effects table, filtered on detector and outlet
            but **not** on period: the trajectory spans the boundary by construction.
        filters (Filters): The active filter state.

    """
    rows = effects[
        effects['scope'].str.fullmatch(r'year_\d{4}')
        & effects['family'].isin(['category', 'pre_year'])
    ].copy()
    if rows.empty:
        _empty('no year rows for the selected detectors.')
        return
    rows['year'] = rows['scope'].str.removeprefix('year_').astype(int)
    figure, axis = theming.figure(width=9.5, height=4.6)
    colours = theming.detector_colours()
    for detector in filters.ordered_detectors():
        series = rows[rows['detector'] == detector].sort_values('year')
        if series.empty:
            continue
        axis.plot(
            series['year'],
            series['delta'],
            marker='o',
            label=theming.DETECTOR_LABEL[detector],
            color=colours[detector],
        )
    axis.axvline(2022.5, color=theming.ink()['muted'], linestyle='--', linewidth=1)
    axis.text(
        2022.6,
        axis.get_ylim()[1] * 0.92,
        'ChatGPT',
        fontsize=8,
        color=theming.ink()['muted'],
    )
    theming.materiality_guides(axis, vertical=False)
    axis.set_xlabel('year')
    axis.set_ylabel("Cliff's δ (counter-consensus vs legacy)")
    axis.legend(fontsize=9)
    theming.show(figure)
    st.caption(
        'The period control does not narrow this panel: the trajectory is only '
        'readable across the boundary.'
    )


def panel_outlet_trend(effects: pd.DataFrame, filters: Filters) -> None:
    """Plot each outlet against its own pre-ChatGPT baseline, year by year.

    Args:
        effects (pd.DataFrame): The effects table, filtered on detector and outlet.
        filters (Filters): The active filter state.

    """
    rows = effects[
        (effects['family'] == 'outlet') & effects['scope'].str.fullmatch(r'year_\d{4}')
    ].copy()
    if rows.empty:
        awaiting(
            'Per-outlet trajectories',
            'python -m calibrate.analyse',
            'Each outlet against its own pre-ChatGPT baseline — the decomposition that '
            'stops the category aggregate being read as adoption.',
        )
        return
    rows['year'] = rows['scope'].str.removeprefix('year_').astype(int)
    choices = filters.ordered_detectors()
    detector = st.selectbox(
        'Detector',
        choices,
        format_func=theming.DETECTOR_LABEL.get,
        key='outlet_trend_detector',
    )
    figure, axis = theming.figure(width=9.5, height=4.6)
    colours = theming.palette()
    for outlet in filters.outlets:
        series = rows[
            (rows['outlet'] == outlet) & (rows['detector'] == detector)
        ].sort_values('year')
        if series.empty:
            continue
        axis.plot(
            series['year'],
            series['delta'],
            marker='o',
            label=theming.OUTLET_LABEL[outlet],
            color=colours[outlet],
        )
    theming.materiality_guides(axis, vertical=False)
    axis.set_xlabel('year')
    axis.set_ylabel("Cliff's δ vs the outlet's own pre-period")
    axis.legend(fontsize=9)
    theming.show(figure)
    st.caption(
        'Within-outlet, so the sampling and curation asymmetries between outlets cannot '
        'contaminate it: the same rules applied to both sides of each contrast.'
    )


def panel_did(effects: pd.DataFrame, filters: Filters) -> None:
    """Show the difference-in-differences table on the 2026 basis.

    Args:
        effects (pd.DataFrame): The effects table, filtered on detector and outlet
            but **not** on period: the contrast needs both sides.
        filters (Filters): The active filter state.

    """
    records = []
    category = effects[effects['family'] == 'category']
    for detector in filters.ordered_detectors():
        rows = category[category['detector'] == detector]
        base = rows[(rows['period'] == 'pre') & (rows['scope'] == 'all')]
        recent = rows[rows['scope'] == 'year_2026']
        if base.empty or recent.empty:
            continue
        pre_delta = float(base.iloc[0]['delta'])
        latest = float(recent.iloc[0]['delta'])
        records.append(
            {
                'detector': theming.DETECTOR_LABEL[detector],
                'PRE δ': round(pre_delta, 3),
                '2026 δ': round(latest, 3),
                'DiD (2026 − PRE)': round(latest - pre_delta, 3),
                'material': abs(latest - pre_delta) >= MATERIALITY,
            }
        )
    if not records:
        _empty(
            'the difference-in-differences needs the category rows for the '
            'selected detectors.'
        )
        return
    st.dataframe(pd.DataFrame(records), width='stretch', hide_index=True)
    st.caption(
        'The difference on record is **2026 δ − PRE δ**, not post minus pre: a '
        'four-year post cell averages a time-varying effect away.'
    )


def panel_sensitivity(effects: pd.DataFrame, filters: Filters) -> None:
    """Show the non-wire sensitivity rows against their headline counterparts.

    The sensitivity family carries both a ``pre`` and a ``post`` row for the same
    scope, so the join onto the headline must key on period as well as scope --
    matching on scope alone silently compares the pre-period non-wire estimate against
    the post-period headline.

    Args:
        effects (pd.DataFrame): The effects table, filtered on detector and outlet.
        filters (Filters): The active filter state.

    """
    rows = effects[effects['family'] == 'sensitivity'].copy()
    if rows.empty:
        _empty('no sensitivity rows for the selected detectors.')
        return
    headline = effects[effects['family'] == 'category'].set_index(
        ['detector', 'period', 'scope']
    )['delta']
    keys = list(zip(rows['detector'], rows['period'], rows['scope'], strict=True))
    rows['headline'] = [headline.get(key, np.nan) for key in keys]
    display = pd.DataFrame(
        {
            'detector': rows['detector'].map(theming.DETECTOR_LABEL),
            'period': rows['period'],
            'scope': rows['scope'],
            'δ (wire excluded)': rows['delta'].round(3),
            'δ (all articles)': rows['headline'].round(3),
            'shift': (rows['delta'] - rows['headline']).round(3),
            'n': rows['n_treatment'] + rows['n_control'],
        }
    ).sort_values(['detector', 'period', 'scope'])
    st.dataframe(display, width='stretch', hide_index=True)

    # The comparison that matters is whether the headline difference-in-differences
    # survives wire exclusion, not whether any single cell moves.
    records = []
    for detector in filters.ordered_detectors():
        cells = rows[rows['detector'] == detector].set_index(['period', 'scope'])
        if ('pre', 'all') not in cells.index or ('post', 'year_2026') not in cells.index:
            continue
        pre = float(cells.loc[('pre', 'all'), 'delta'])
        recent = float(cells.loc[('post', 'year_2026'), 'delta'])
        head_pre = headline.get((detector, 'pre', 'all'), np.nan)
        head_recent = headline.get((detector, 'post', 'year_2026'), np.nan)
        records.append(
            {
                'detector': theming.DETECTOR_LABEL[detector],
                'DiD (wire excluded)': round(recent - pre, 3),
                'DiD (all articles)': round(head_recent - head_pre, 3),
                'still material': abs(recent - pre) >= MATERIALITY,
            }
        )
    if records:
        st.dataframe(pd.DataFrame(records), width='stretch', hide_index=True)
    st.caption(
        'Wire copy is flagged and reported, never dropped from the headline — that is '
        'one of the three mitigations committed to in the Interim Report. RTÉ runs 34% '
        'wire and the Examiner 18%, so this bounds how much of the contrast agency '
        'copy could be carrying. The second table is the test that matters: whether '
        'the difference-in-differences survives wire exclusion.'
    )


# ---------- CALIBRATION ----------
def panel_thresholds(calibration: pd.DataFrame, filters: Filters) -> None:
    """Show the fitted per-outlet thresholds and the FPR they actually realised.

    Args:
        calibration (pd.DataFrame): The calibration report, already filtered.
        filters (Filters): The active filter state.

    """
    if calibration.empty:
        _empty('no calibration rows for the selected detectors and outlets.')
        return
    display = pd.DataFrame(
        {
            'detector': calibration['detector'].map(theming.DETECTOR_LABEL),
            'outlet': calibration['outlet'].map(theming.OUTLET_LABEL),
            'threshold': calibration['threshold'].round(4),
            'n (anchor)': calibration['n_human'],
            'FPR realised': calibration['fpr_realised'].round(4),
            'TPR (all generators)': calibration['tpr'].round(4),
        }
    )
    st.dataframe(display, width='stretch', hide_index=True)
    st.caption(
        f'Thresholds fitted per outlet on its held-out human anchor at a '
        f'{filters.fpr_target:.0%} target, then applied unchanged to the corpus. '
        'Fitting per outlet is what stops a house style being read as AI.'
    )


def panel_band_tpr(bands: pd.DataFrame, filters: Filters) -> None:
    """Show detector sensitivity by article length, which is the power picture.

    Args:
        bands (pd.DataFrame): ``tpr_by_band.csv``, already filtered.
        filters (Filters): The active filter state.

    """
    rows = bands[bands['estimable']].copy()
    if rows.empty:
        _empty('no estimable length bands for this selection.')
        return
    order = ['0-150', '150-300', '300-600', '600-inf']
    present = [band for band in order if band in set(rows['band'])]
    grid = rows.pivot_table(
        index='detector', columns='band', values='tpr', aggfunc='mean'
    ).reindex(columns=present)
    grid = grid.reindex([d for d in filters.ordered_detectors() if d in grid.index])
    grid.index = [theming.DETECTOR_LABEL.get(name, name) for name in grid.index]

    figure, axis = theming.figure(width=8.5, height=0.6 * len(grid) + 1.8)
    image = axis.imshow(
        grid.to_numpy(), cmap=theming.heatmap_cmap(), vmin=0, vmax=1, aspect='auto'
    )
    axis.set_xticks(range(grid.shape[1]))
    axis.set_xticklabels(grid.columns, fontsize=9)
    axis.set_yticks(range(grid.shape[0]))
    axis.set_yticklabels(grid.index, fontsize=9)
    for row_index in range(grid.shape[0]):
        for column_index in range(grid.shape[1]):
            value = grid.iloc[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                '—' if pd.isna(value) else f'{value:.2f}',
                ha='center',
                va='center',
                fontsize=8,
                color='white' if pd.isna(value) or value < HEATMAP_INK_FLIP else 'black',
            )
    figure.colorbar(image, ax=axis, label=f'TPR at {filters.fpr_target:.0%} FPR')
    axis.set_xlabel('article length (words)')
    theming.show(figure)

    dropped = bands[~bands['estimable']]
    if not dropped.empty:
        smallest = int(dropped['n_reference'].min())
        st.caption(
            f'Warning: {len(dropped)} outlet × band cells are not estimable and are blank '
            f'above — the thinnest holds {smallest} reference articles. The 0–150 word '
            'band is the one that runs out: short text carries too little signal for '
            'any of these detectors, which is exactly why the length-band co-primary '
            'exists.'
        )


def panel_rules(rules: pd.DataFrame, filters: Filters) -> None:
    """Compare the candidate ensemble decision rules on the same footing.

    Args:
        rules (pd.DataFrame): ``rule_comparison_*.csv``, already filtered.
        filters (Filters): The active filter state.

    """
    if rules.empty:
        _empty('no rule rows for the selected outlets and period.')
        return
    summary = (
        rules.groupby('rule')
        .agg(
            anchor_fpr=('anchor_fpr', 'mean'),
            tpr_all=('tpr_all', 'mean'),
            tpr_frontier=('tpr_frontier', 'mean'),
            detected=('detected', 'mean'),
            lower_bound=('lower_bound', 'mean'),
        )
        .reset_index()
        .sort_values('tpr_frontier', ascending=False)
    )
    display = summary.rename(
        columns={
            'rule': 'rule',
            'anchor_fpr': 'FPR (anchor)',
            'tpr_all': 'TPR (all generators)',
            'tpr_frontier': 'TPR (frontier)',
            'detected': 'detected rate',
            'lower_bound': 'lower bound',
        }
    ).round(4)
    st.dataframe(display, width='stretch', hide_index=True)
    st.caption(
        f'Averaged over the selected outlets and periods at a '
        f'{filters.fpr_target:.0%} target. **Read the frontier column, not the "all '
        'generators" column.** Every rule looks strong against open-weight text and '
        'every rule collapses against frontier text; a rule chosen on the aggregate '
        'would be chosen on the easy cases.'
    )


def panel_sanity(sanity: pd.DataFrame) -> None:
    """Show the pre-period false-positive sanity check against target.

    Args:
        sanity (pd.DataFrame): The sanity table, already filtered.

    """
    if sanity.empty:
        _empty('no sanity rows for the selected outlets.')
        return
    rows = sanity.copy()
    rows['outlet'] = rows['outlet'].map(theming.OUTLET_LABEL).fillna(rows['outlet'])
    st.dataframe(rows.round(4), width='stretch', hide_index=True)
    st.caption(
        'Every pre-period article predates ChatGPT, so the realised detected rate '
        'there is a false-positive rate. All four land below target, i.e. the '
        'thresholds are conservative — which is the same thing as saying every '
        'prevalence figure downstream is a floor.'
    )


# ---------- PREVALENCE ----------


def panel_inversion(calibration: pd.DataFrame, filters: Filters) -> None:
    """Plot per-generator sensitivity, where the frontier model breaks the pattern.

    Args:
        calibration (pd.DataFrame): The calibration report, already filtered.
        filters (Filters): The active filter state.

    """
    columns = [column for column in calibration.columns if column.startswith('tpr_')]
    if not columns or calibration.empty:
        _empty('the calibration report carries no per-generator TPR columns here.')
        return
    means = calibration.groupby('detector')[columns].mean()
    order = [d for d in filters.ordered_detectors() if d in means.index]
    means = means.reindex(order)
    means.index = [theming.DETECTOR_LABEL.get(name, name) for name in means.index]
    means.columns = [column.removeprefix('tpr_') for column in means.columns]

    figure, axis = theming.figure(width=9.5, height=0.6 * len(means) + 1.8)
    image = axis.imshow(
        means.to_numpy(), cmap=theming.heatmap_cmap(), vmin=0, vmax=1, aspect='auto'
    )
    axis.set_xticks(range(len(means.columns)))
    axis.set_xticklabels(means.columns, rotation=30, ha='right', fontsize=9)
    axis.set_yticks(range(len(means.index)))
    axis.set_yticklabels(means.index, fontsize=9)
    for row_index in range(means.shape[0]):
        for column_index in range(means.shape[1]):
            value = means.iloc[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f'{value:.2f}',
                ha='center',
                va='center',
                fontsize=8,
                color='white' if value < HEATMAP_INK_FLIP else 'black',
            )
    figure.colorbar(image, ax=axis, label=f'TPR at {filters.fpr_target:.0%} FPR')
    theming.show(figure)
    st.caption(
        'The GPT-4o column is the finding: the curvature detectors collapse on it '
        'while holding 0.82–0.93 against open-weight models. Perplexity does not. '
        'Averaged over the selected outlets.'
    )


# ---------- GENERATION ----------
def _ladder_frame() -> pd.DataFrame | None:
    """Load and concatenate the two generation manifests, without their text.

    Returns:
        pd.DataFrame | None: The combined manifest, or ``None`` if neither exists.

    """
    parts = [
        frame
        for frame in (
            read(GENERATED_IRISH, LADDER_COLUMNS),
            read(GENERATED_FRONTIER, LADDER_COLUMNS),
        )
        if frame is not None
    ]
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def panel_ladder(generated: pd.DataFrame, filters: Filters) -> None:
    """Show the Irish-register generation ladder and where it is incomplete.

    Args:
        generated (pd.DataFrame): The combined generation manifest.
        filters (Filters): The active filter state.

    """
    total = len(generated)
    target = LADDER_TARGET_PER_MODEL * len(LADDER_MODELS)
    columns = st.columns(3)
    columns[0].metric('Generated', f'{total:,}')
    columns[1].metric(f'Target ({len(LADDER_MODELS)} models)', f'{target:,}')
    columns[2].metric('Complete', f'{total / target:.0%}')

    scoped = generated[generated['outlet'].isin(filters.outlets)]
    if scoped.empty:
        _empty('the ladder has no rows for the selected outlets.')
        return
    grid = scoped.pivot_table(
        index='model', columns='outlet', aggfunc='size', fill_value=0
    )
    grid = grid.reindex(index=[m for m in LADDER_MODELS if m in grid.index])
    grid = grid.reindex(columns=[o for o in theming.OUTLETS if o in grid.columns])
    grid.columns = theming.label_outlets(grid.columns)

    per_outlet_target = LADDER_TARGET_PER_MODEL / len(theming.OUTLETS)
    figure, axis = theming.figure(width=8.5, height=0.5 * len(grid) + 1.8)
    image = axis.imshow(
        grid.to_numpy(),
        cmap=theming.heatmap_cmap(),
        vmin=0,
        vmax=per_outlet_target,
        aspect='auto',
    )
    axis.set_xticks(range(grid.shape[1]))
    axis.set_xticklabels(grid.columns, rotation=20, ha='right', fontsize=9)
    axis.set_yticks(range(grid.shape[0]))
    axis.set_yticklabels(grid.index, fontsize=9)
    for row_index in range(grid.shape[0]):
        for column_index in range(grid.shape[1]):
            value = grid.iloc[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                str(value),
                ha='center',
                va='center',
                fontsize=8,
                color='black' if value > per_outlet_target * 0.6 else 'white',
            )
    figure.colorbar(image, ax=axis, label=f'articles (target {per_outlet_target:.0f})')
    theming.show(figure)

    if 'era' in scoped.columns:
        st.bar_chart(scoped.groupby('era').size().rename('articles'))
        st.caption(
            'Model era. The open-weight rungs run 2019–2024; the 2025 and 2026 bars '
            'are the frontier rungs, which is what lets recency be separated from '
            'proprietary post-training.'
        )
    st.caption(
        'Empty cells are generation gaps, not scoring gaps — every article that exists '
        'has been scored by all four detectors.'
    )


# ---------- CLASSIFIER ----------
CLASSIFIER_TARGET_F1: Final = 0.85
# Recall bands the intervals actually separate. Point estimates order all seven
# generators, but at n~600 only these three groups are distinguishable.
GENERATOR_GROUPS: Final = {
    'gpt2-large': 'base',
    'gptneo-1.3b': 'base',
    'llama2-7b': 'instruct',
    'mistral-7b': 'instruct',
    'qwen2.5-7b': 'instruct',
    'gpt-4.1': 'frontier',
    'claude-opus-4-6': 'frontier',
}
GROUP_COLOURS: Final = {'base': '#d98a2b', 'instruct': '#4a9de0', 'frontier': '#e06c6c'}


def _recall_chart(rows: pd.DataFrame) -> None:
    """Plot per-generator recall with confidence intervals where available.

    Args:
        rows (pd.DataFrame): Rows carrying ``generator``, ``rate`` and optionally
            ``ci_low`` / ``ci_high``.

    """
    rows = rows.assign(group=rows['generator'].map(GENERATOR_GROUPS).fillna('other'))
    rows = rows.sort_values(['group', 'rate'])
    figure, axis = theming.figure(width=7, height=0.45 * len(rows) + 1.4)

    has_ci = {'ci_low', 'ci_high'}.issubset(rows.columns) and rows['ci_low'].notna().any()
    errors = None
    if has_ci:
        errors = [
            (rows['rate'] - rows['ci_low']).to_numpy(),
            (rows['ci_high'] - rows['rate']).to_numpy(),
        ]
    axis.barh(
        rows['generator'],
        rows['rate'],
        xerr=errors,
        color=[GROUP_COLOURS.get(group, '#6b7789') for group in rows['group']],
        error_kw={'ecolor': theming.ink()['text'], 'capsize': 3, 'lw': 1},
    )
    axis.set_xlim(0, 1.02)
    axis.set_xlabel('recall (AI class)')
    for group, colour in GROUP_COLOURS.items():
        axis.scatter([], [], color=colour, label=group)
    axis.legend(loc='lower left', fontsize=8, frameon=False)
    theming.show(figure)


def panel_classifier(report: pd.DataFrame | None) -> None:
    """Show the supervised classifier results, or the placeholder until they exist.

    Args:
        report (pd.DataFrame | None): The classifier report, if it exists.

    """
    if report is None:
        awaiting(
            'Supervised classifier',
            'python -m classify.dataset\npython -m classify.baseline\n'
            'python -m classify.crossval',
            "The Proposal's third objective: F1 ≥ 0.85 on a held-out split of the "
            'Irish anchor against the Irish-register generated set, per-generator '
            'recall, and the false-positive rate on human articles.',
        )
        return

    if 'model' in report.columns:
        chosen = st.segmented_control(
            'Model',
            sorted(report['model'].unique()),
            default=min(report['model'].unique()),
            key='classifier_model',
            help='tfidf is the conservative instrument; frozen is the more sensitive '
            'one, at roughly double the false-positive rate.',
        )
        if chosen:
            report = report[report['model'] == chosen]

    rate = 'rate' if 'rate' in report.columns else 'recall'
    scope = report['scope'].fillna('')
    overall = report[scope == 'overall']
    if not overall.empty:
        row = overall.iloc[0]
        f1 = row['rate'] if 'rate' in report.columns else row['f1']
        columns = st.columns(4)
        columns[0].metric('F1 (AI class)', f'{f1:.3f}')
        columns[1].metric('Precision', f'{row["precision"]:.3f}')
        columns[2].metric('Recall', f'{row["recall"]:.3f}')
        columns[3].metric(
            f'Target {CLASSIFIER_TARGET_F1}',
            'met' if f1 >= CLASSIFIER_TARGET_F1 else 'not met',
        )

    recall_rows = report[scope.str.startswith('recall:')].copy()
    if not recall_rows.empty:
        recall_rows['generator'] = recall_rows['scope'].str.removeprefix('recall:')
        recall_rows = recall_rows.rename(columns={rate: 'rate'})
        _recall_chart(recall_rows)
        st.caption(
            'Per-generator recall, grouped by the contrasts the intervals actually '
            'separate. **Base is separated from instruct, and frontier from instruct; '
            'the three instruct rungs are not distinguishable from each other, and '
            'neither are the two frontier rungs.** Read the within-group ordering as an '
            'observation, not a gradient. Note this is the mirror image of the zero-shot '
            'detectors, which find base models trivial and invert on frontier output.'
        )

    false_positive = report[scope.str.startswith('false_positive_rate')]
    if not false_positive.empty:
        value = float(false_positive.iloc[0][rate])
        st.warning(
            f'False-positive rate on human articles: **{value:.1%}** — roughly one in '
            f'{round(1 / value) if value else 0} genuine articles is flagged. This model '
            'is trained on Irish news against seven specific generators and is not a '
            'general-purpose AI detector.'
        )
    st.dataframe(report.round(4), width='stretch', hide_index=True)


# ---------- TIME SERIES ----------
def panel_score_over_time(scored: pd.DataFrame, filters: Filters) -> None:
    """Plot median detector score per outlet over time, from the scored corpus.

    Args:
        scored (pd.DataFrame): The scored corpus metadata, already filtered.
        filters (Filters): The active filter state.

    """
    detector = st.selectbox(
        'Detector',
        filters.ordered_detectors(),
        format_func=theming.DETECTOR_LABEL.get,
        key='time_series_detector',
    )
    column = f'{detector}_score'
    if column not in scored.columns or 'published_date' not in scored.columns:
        st.info('The scored corpus does not carry that column.')
        return
    frame = scored[['outlet', 'published_date', column]].dropna().copy()
    if frame.empty:
        _empty('no scored articles survive the current selection.')
        return
    frame['date'] = pd.to_datetime(frame['published_date'], errors='coerce')
    frame = frame.dropna(subset=['date'])
    frame['quarter'] = frame['date'].dt.to_period('Q').dt.to_timestamp()
    series = frame.pivot_table(
        index='quarter', columns='outlet', values=column, aggfunc='median'
    )
    series = series.reindex(columns=[o for o in theming.OUTLETS if o in series.columns])

    figure, axis = theming.figure(width=10, height=4.4)
    colours = theming.palette()
    for outlet in series.columns:
        axis.plot(
            series.index,
            series[outlet],
            label=theming.OUTLET_LABEL[outlet],
            color=colours[outlet],
            linewidth=1.6,
        )
    axis.axvline(
        pd.Timestamp('2022-11-30'),
        color=theming.ink()['muted'],
        linestyle='--',
        linewidth=1,
    )
    axis.set_ylabel(f'median {theming.DETECTOR_LABEL[detector]} score')
    axis.set_xlabel('quarter')
    axis.legend(fontsize=9)
    theming.show(figure)
    st.caption(
        'Raw detector scores, not calibrated probabilities, and not comparable across '
        'detectors. The dashed line is the ChatGPT release. Unlike the reported '
        'tables, this panel is computed live, so the length and wire filters bind.'
    )


# ---------- PAGES ----------
def page_results(filters: Filters) -> None:
    """Render the effect-size page.

    Args:
        filters (Filters): The active filter state.

    """
    effects = read(PRIMARY_EFFECTS)
    if effects is None:
        awaiting(
            'Effect sizes',
            'python -m calibrate.calibrate\npython -m calibrate.analyse',
            'The pre-specified primary, the length-band co-primary, the PRE placebo, '
            'the year trajectory and the per-outlet decomposition.',
        )
        return
    spanning = filters.by_outlet(filters.by_detector(effects))
    scoped = filters.by_period(spanning)

    st.subheader('The pre-specified primary')
    panel_primary(scoped)
    st.divider()
    st.subheader('Every category contrast')
    panel_forest(scoped)
    st.divider()
    st.subheader('Co-primary — by length band')
    panel_bands(spanning, filters)
    st.divider()
    st.subheader('Year by year')
    panel_year_trend(spanning, filters)
    panel_did(spanning, filters)
    st.divider()
    st.subheader('Each outlet against its own baseline')
    panel_outlet_trend(spanning, filters)
    st.divider()
    st.subheader('Sensitivity — wire copy excluded')
    panel_sensitivity(spanning, filters)


def page_calibration(filters: Filters) -> None:
    """Render the calibration page: thresholds, sensitivity by length, decision rules.

    Args:
        filters (Filters): The active filter state.

    """
    calibration = read(CALIBRATION_REPORT)
    if calibration is None:
        awaiting(
            'Calibration',
            'python -m calibrate.calibrate',
            'Per-outlet thresholds fitted on the held-out human anchor, the '
            'false-positive rate they realise, and the sensitivity they buy.',
        )
    else:
        st.subheader('Thresholds and realised false-positive rate')
        panel_thresholds(filters.pipeline(calibration), filters)
        st.divider()

    sanity = read_for_fpr(SANITY_STEM, filters)
    if sanity is not None:
        st.subheader('Pre-period false-positive check')
        panel_sanity(filters.pipeline(sanity))
        st.divider()

    bands = read(BAND_TPR)
    if bands is None:
        awaiting(
            'Sensitivity by article length',
            'python -m calibrate.analyse',
            'Where the design has power and where it does not: detector recall '
            'stratified by article length.',
        )
    else:
        st.subheader('Sensitivity by article length')
        panel_band_tpr(filters.pipeline(bands), filters)
        st.divider()

    rules = read_for_fpr(RULES_STEM, filters)
    if rules is None:
        awaiting(
            'Ensemble decision rules',
            'python -m calibrate.calibrate',
            'The candidate combination rules — majority vote, any-two, single '
            'detector — scored on the same anchor so the choice is evidenced.',
        )
    else:
        st.subheader('Ensemble decision rules')
        panel_rules(filters.pipeline(rules), filters)


def page_prevalence(filters: Filters) -> None:
    """Render the prevalence and detector-capability page.

    Args:
        filters (Filters): The active filter state.

    """
    calibration = read(CALIBRATION_REPORT)
    if calibration is not None:
        st.subheader('Sensitivity per generator — where the design runs out of power')
        panel_inversion(filters.pipeline(calibration), filters)


def page_generation(filters: Filters) -> None:
    """Render the generation-ladder page.

    Args:
        filters (Filters): The active filter state.

    """
    generated = _ladder_frame()
    if generated is None:
        awaiting(
            'Irish-register generation ladder',
            'jupyter notebook detect/Detect-optional/colab_generate_irish_ai_v3.ipynb',
            'Register-matched AI text used to calibrate what a human-to-AI shift looks '
            'like in Irish news prose.',
        )
        return
    st.subheader('Ladder coverage')
    panel_ladder(generated, filters)


def page_classifier(_filters: Filters) -> None:
    """Render the supervised-classifier page.

    Args:
        _filters (Filters): Unused — the classifier reports carry no outlet, period or
            detector dimension, so nothing in the sidebar can narrow them.

    """
    st.subheader('Supervised classifier')
    st.caption(
        'The sidebar does not reach this page: the cross-validated report is written '
        'per model and per generator, with no outlet, period or detector dimension.'
    )
    report = next((r for r in map(read, CLASSIFIER_REPORTS) if r is not None), None)
    panel_classifier(report)


def page_time_series(scored: pd.DataFrame, filters: Filters) -> None:
    """Render the time-series page.

    Args:
        scored (pd.DataFrame): The filtered scored-corpus metadata.
        filters (Filters): The active filter state.

    """
    st.subheader('Detector score over time')
    panel_score_over_time(scored, filters)
