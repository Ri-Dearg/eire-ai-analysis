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
