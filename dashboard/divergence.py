# ruff: noqa: RUF001  (en dashes and minus signs are intentional in labels)
"""Divergence page: trajectory, length control, drift control, step test, production."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import pandas as pd
import streamlit as st

from dashboard import theming

if TYPE_CHECKING:
    from dashboard.filters import Filters

ROOT = Path(__file__).resolve().parent.parent
PREVALENCE = ROOT / 'data' / 'classify' / 'corpus_prevalence_report.csv'
PRODUCTION = ROOT / 'data' / 'detection' / 'production_evidence.csv'

LAST_AI_FREE_YEAR: Final = 2022
BAND_ORDER: Final = ('0-150', '150-300', '300-600', '600-inf')


@st.cache_data(show_spinner=False)
def _load(path: Path) -> pd.DataFrame | None:
    """Return a CSV as a frame, or ``None`` if absent.

    Args:
        path (Path): File to read.

    Returns:
        pd.DataFrame | None: The table, or ``None``.

    """
    return pd.read_csv(path) if path.exists() else None


def _years(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the rows whose ``year`` is a calendar year, typed as an int.

    Args:
        frame (pd.DataFrame): Rows from the prevalence report.

    Returns:
        pd.DataFrame: The calendar-year rows, with an integer ``year``.

    """
    out = frame.copy()
    out['year'] = pd.to_numeric(out['year'], errors='coerce')
    return out.dropna(subset=['year']).astype({'year': int})


def _scope(report: pd.DataFrame, scope: str, filters: Filters) -> pd.DataFrame:
    """Select one scope from the prevalence report, honouring the sidebar.

    Args:
        report (pd.DataFrame): ``corpus_prevalence_report.csv``.
        scope (str): The scope to select.
        filters (Filters): The active filter state.

    Returns:
        pd.DataFrame: The selected rows.

    """
    rows = report[
        (report['scope'] == scope) & (report['fpr_target'] == filters.fpr_target)
    ]
    return rows[rows['outlet'].isin(filters.outlets)]


def panel_trajectory(report: pd.DataFrame, filters: Filters) -> None:
    """Plot each outlet's flag rate by year against a threshold fixed on 2019.

    Args:
        report (pd.DataFrame): ``corpus_prevalence_report.csv``.
        filters (Filters): The active filter state.

    """
    st.markdown("#### Divergence from each outlet's own 2019 prose")
    st.caption(
        'Threshold fitted per outlet on its 2019 articles at a '
        f'{filters.fpr_target:.0%} false-positive rate, then every later year scored '
        'against that fixed bar. **2020–2022 predate ChatGPT**, so movement there is '
        "the outlet's own rate of change with the technology absent."
    )

    rows = _scope(report, 'trajectory', filters)
    if rows.empty:
        st.info('No trajectory rows — run `python -m classify.corpus_prevalence`.')
        return

    figure, axis = theming.figure(width=7.5, height=4.2)
    shade = theming.ink()
    axis.axvspan(2019, LAST_AI_FREE_YEAR, color=shade['grid'], zorder=0)
    colours = theming.palette()
    for outlet in filters.outlets:
        cell = _years(rows[rows['outlet'] == outlet]).sort_values('year')
        if cell.empty:
            continue
        counter = theming.CATEGORY[outlet] == 'counter-consensus'
        axis.plot(
            cell['year'],
            cell['value'] * 100,
            marker='o',
            markersize=4,
            linewidth=2 if counter else 1.4,
            linestyle='-' if counter else '--',
            color=colours[outlet],
            label=f'{theming.OUTLET_LABEL[outlet]} ({theming.CATEGORY[outlet]})',
        )
    axis.text(
        2019.1,
        axis.get_ylim()[1],
        ' pre-ChatGPT',
        va='top',
        fontsize=8,
        color=shade['muted'],
    )
    axis.set_xlabel('year')
    axis.set_ylabel('articles flagged (%)')
    axis.legend(fontsize=8)
    theming.show(figure)

    st.caption(
        '**Read the shaded band first.** Three of the four outlets are flat across it. '
        'A rise after 2022 is only interpretable because the baseline is not rising.'
    )


def panel_trajectory_bands(report: pd.DataFrame, filters: Filters) -> None:
    """Repeat the trajectory inside fixed length bands, as a length control.

    Args:
        report (pd.DataFrame): ``corpus_prevalence_report.csv``.
        filters (Filters): The active filter state.

    """
    st.markdown('#### The same trajectory, inside fixed length bands')
    st.caption(
        'The counter-consensus outlets also got longer across the window, and length '
        'moves classifier scores on its own. If the divergence were a length artefact '
        'it would flatten here. It does not.'
    )

    rows = _scope(report, 'trajectory_band', filters)
    if rows.empty:
        st.info('No length-band rows — re-run `python -m classify.corpus_prevalence`.')
        return

    bands = [band for band in BAND_ORDER if band in set(rows['band'])]
    band = st.segmented_control(
        'Length band (words)',
        bands,
        default=bands[-2] if len(bands) > 1 else bands[0],
        key='divergence_band',
    )
    if not band:
        band = bands[0]
    cells = _years(rows[rows['band'] == band])
    if cells.empty:
        st.info('That band holds no rows for the selected outlets.')
        return

    figure, axis = theming.figure(width=7.5, height=4.0)
    shade = theming.ink()
    axis.axvspan(2019, LAST_AI_FREE_YEAR, color=shade['grid'], zorder=0)
    colours = theming.palette()
    thin = []
    for outlet in filters.outlets:
        cell = cells[cells['outlet'] == outlet].sort_values('year')
        if cell.empty:
            continue
        if int(cell['n'].min()) < 100:  # noqa: PLR2004
            thin.append(theming.OUTLET_LABEL[outlet])
        axis.plot(
            cell['year'],
            cell['value'] * 100,
            marker='o',
            markersize=4,
            linewidth=2 if theming.CATEGORY[outlet] == 'counter-consensus' else 1.4,
            linestyle='-' if theming.CATEGORY[outlet] == 'counter-consensus' else '--',
            color=colours[outlet],
            label=theming.OUTLET_LABEL[outlet],
        )
    axis.set_xlabel('year')
    axis.set_ylabel(f'articles flagged, {band} words (%)')
    axis.legend(fontsize=8)
    theming.show(figure)
    if thin:
        st.caption(
            f'⚠️ Fewer than 100 articles in at least one year for: {", ".join(thin)}. '
            'Read those lines as indicative.'
        )


def panel_drift_control(report: pd.DataFrame, filters: Filters) -> None:
    """Show each outlet's fitted AI-free drift against its measured 2026 rate.

    Args:
        report (pd.DataFrame): ``corpus_prevalence_report.csv``.
        filters (Filters): The active filter state.

    """
    st.markdown('#### The drift control')
    st.caption(
        'A change in house style is **not** a rival explanation to AI adoption — where '
        'AI is adopted, a change in style is what you would expect to see. What this '
        "measures is each outlet's *capacity* for change with the technology absent."
    )

    slope = _scope(report, 'drift_slope', filters).set_index('outlet')['value']
    predicted = _scope(report, 'drift_extrapolated_2026', filters)
    predicted = predicted.set_index('outlet')['value']
    trajectory = _years(_scope(report, 'trajectory', filters))
    actual = trajectory[trajectory['year'] == 2026].set_index('outlet')['value']  # noqa: PLR2004
    if slope.empty:
        st.info('No drift rows — run `python -m classify.corpus_prevalence`.')
        return

    table = pd.DataFrame(
        {
            'Outlet': theming.label_outlets(slope.index),
            'Category': [theming.CATEGORY[o] for o in slope.index],
            'Drift, pts/year (2019–22)': [f'{slope[o] * 100:+.2f}' for o in slope.index],
            'Predicted 2026': [
                f'{max(predicted.get(o, 0), 0) * 100:.1f}%' for o in slope.index
            ],
            'Measured 2026': [
                f'{actual.get(o, float("nan")) * 100:.1f}%' for o in slope.index
            ],
            'Excess': [
                f'{(actual.get(o, 0) - predicted.get(o, 0)) * 100:+.1f} pts'
                for o in slope.index
            ],
        }
    ).sort_values('Measured 2026', ascending=False)
    st.dataframe(table, hide_index=True, width='stretch')

    placebo = _years(_scope(report, 'drift_placebo', filters))
    if not placebo.empty:
        wide = placebo.pivot_table(index='outlet', columns='year', values='value')
        wide.index = theming.label_outlets(wide.index)
        with st.expander('Placebo: the same extrapolation aimed at a pre-ChatGPT year'):
            st.dataframe((wide * 100).round(2), width='stretch')
            st.caption(
                'The drift fit is re-run holding out a year that predates ChatGPT, so '
                'the predicted value can be checked against a known answer. These land '
                'near 1%, which is the target false-positive rate — i.e. the '
                'extrapolation is not manufacturing an excess on its own.'
            )

    st.caption(
        'Drift is fitted by least squares on the four pre-ChatGPT years only. '
        "Warning: Gript's 2019 cell holds 388 articles, so its anchor point is thin; its "
        '2020–22 points sit on 1,211–2,662 and are flat, so the flat baseline holds.'
    )


def panel_step(report: pd.DataFrame, filters: Filters) -> None:
    """Show year-on-year change, which separates a step from a ramp.

    Args:
        report (pd.DataFrame): ``corpus_prevalence_report.csv``.
        filters (Filters): The active filter state.

    """
    st.markdown('#### Step or ramp?')
    rows = _years(_scope(report, 'trajectory', filters))
    if rows.empty:
        return
    wide = rows.pivot_table(index='outlet', columns='year', values='value')
    delta = (wide.diff(axis=1) * 100).round(1).dropna(axis=1, how='all')
    delta.index = theming.label_outlets(delta.index)
    st.dataframe(
        delta.style.format('{:+.1f}').background_gradient(cmap='Reds', vmin=0, vmax=20),
        width='stretch',
    )
    st.caption(
        '**The Liberal moves +19.4 points in 2024→2025** — four to five times any '
        'year-on-year change it has ever made. **Gript is the other shape**: a steady '
        'ramp from a flat base, with no step. The legacy outlets do neither.'
    )


def panel_production(production: pd.DataFrame, filters: Filters) -> None:
    """Show byline concentration and implied workload, without naming anyone.

    Args:
        production (pd.DataFrame): ``production_evidence.csv``.
        filters (Filters): The active filter state.

    """
    st.markdown('#### Production evidence — no detector involved')
    st.caption(
        'Byline counts and article lengths from publication metadata, joined to true '
        'output volume from the crawl inventory. Individuals are never named: the unit '
        'of analysis is the outlet.'
    )

    outlet = st.selectbox(
        'Outlet',
        filters.outlets,
        format_func=lambda slug: theming.OUTLET_LABEL[slug],
        key='prod_outlet',
    )
    rows = production[production['outlet'] == outlet].sort_values('year')
    if rows.empty:
        st.info('No rows — run `python -m calibrate.production`.')
        return

    display = pd.DataFrame(
        {
            'Year': rows['year'],
            'Distinct bylines': rows['n_bylines'],
            'Top-2 share': (rows['top2_share'] * 100).round(0).astype(int).astype(str)
            + '%',
            'Median words': rows['median_words'].round(0).astype(int),
            'Published': rows['published'].astype('Int64'),
            'Implied words/day, busiest byline': [
                '—' if pd.isna(value) else f'{value:,.0f}'
                for value in rows['implied_words_per_day']
            ],
        }
    )
    st.dataframe(display, hide_index=True, width='stretch')

    if bool(rows['top1_is_desk'].all()):
        st.caption(
            f'Warning: {theming.OUTLET_LABEL[outlet]} credits its highest-volume output to a '
            'desk (e.g. “RTÉ News”, “Gript News”, “Digital Desk staff”), so a '
            'per-person workload figure would describe an organisation and is not '
            'shown.'
        )
    if outlet == 'the_liberal':
        st.warning(
            'The Liberal is the only outlet attributing its highest-volume output to '
            'named individuals throughout. Distinct bylines fall **7 → 2** while median '
            "length rises **214 → 398 words**, taking the busiest byline's implied "
            'output to roughly **3,400 words per calendar day** in 2026. The step is '
            '2024→2025 — the same year the text measure steps.'
        )
    if outlet == 'gript':
        st.info(
            "Gript's production profile is stable and ordinary: 29–37 bylines "
            'throughout, top-2 share 27–35%, busiest named byline around 1,000–1,270 '
            'words/day with no step. **Its text divergence is real but is not '
            'corroborated by production evidence** — unlike The Liberal, where the two '
            'converge.'
        )
    if outlet == 'irish_examiner':
        st.error(
            'Data defect: **all 1,398 Irish Examiner articles from 2023 are '
            'unattributed**, against 529 distinct bylines in 2022 and 531 in 2024. '
            'This is a year-specific parse failure, not a change in practice. Byline '
            'statistics for the Examiner should exclude 2023.'
        )


def panel_caveat() -> None:
    """State plainly what the divergence finding does and does not establish."""
    with st.container(border=True):
        st.markdown('#### What this does and does not show')
        st.markdown(
            '- **Established:** the ordering, and the timing. Two counter-consensus '
            'outlets diverge sharply from their own pre-ChatGPT prose; two legacy '
            'outlets do not, against flat baselines.\n'
            "- **Not established:** the cause, or the level. The classifier's human "
            'class is drawn entirely from 2019–2022, so its false-positive rate on '
            'post-2022 human writing is unmeasured — which is why these are flag rates '
            'and not prevalence estimates.\n'
            '- **Also worth knowing:** “pre-ChatGPT” is not “pre-LLM”. GPT-3 API access '
            'dates from 2020, so the baseline window is ChatGPT-free but not '
            'necessarily assistance-free. That biases toward understating any effect.'
        )


def page_divergence(filters: Filters) -> None:
    """Render the divergence page.

    Args:
        filters (Filters): The active filter state.

    """
    st.subheader('Divergence from baseline')
    report = _load(PREVALENCE)
    production = _load(PRODUCTION)

    if report is None:
        st.info(
            'Run `python -m classify.corpus_prevalence` to produce the trajectory, '
            'length-band, drift-control and step panels.'
        )
    else:
        panel_trajectory(report, filters)
        st.divider()
        panel_trajectory_bands(report, filters)
        st.divider()
        panel_drift_control(report, filters)
        st.divider()
        panel_step(report, filters)
        st.divider()

    if production is None:
        st.info('Run `python -m calibrate.production` to produce the production panel.')
    else:
        panel_production(production, filters)

    panel_caveat()
