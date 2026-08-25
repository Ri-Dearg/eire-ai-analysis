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

    The ``year`` column doubles as a period label elsewhere in the same table -- it
    holds values like ``post_all`` -- so it arrives as an object column.

    Args:
        frame (pd.DataFrame): Rows from the prevalence report.

    Returns:
        pd.DataFrame: The calendar-year rows, with an integer ``year``.

    """
    out = frame.copy()
    out['year'] = pd.to_numeric(out['year'], errors='coerce')
    return out.dropna(subset=['year']).astype({'year': int})
