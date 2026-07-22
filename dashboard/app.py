from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
SCORED = ROOT / 'data' / 'corpus_scored.csv'

OUTLETS = ['rte', 'irish_examiner', 'the_liberal', 'gript']

BANNER = (
    'Detector scores are a **lower bound**, not a count. '
    'Pre-ChatGPT articles are a false-positive control. '
    'All processing is local; no article text is loaded.'
)


def sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """Draw the global filter sidebar and return the filtered frame.

    Args:
        df (pd.DataFrame): DataFrame to display.

    Returns:
        pd.DataFrame: Dataframe with sidebar.

    """
    st.sidebar.header('Filters')
    picked = st.sidebar.multiselect(
        'Outlets',
        OUTLETS,
        default=OUTLETS,
    )
    period = st.sidebar.radio('Period', ['both', 'pre', 'post'], horizontal=True)
    min_words = st.sidebar.slider('Minimum word count', 0, 1500, 0, 50)
    drop_wire = st.sidebar.checkbox('Exclude wire copy')
    st.sidebar.divider()
    st.sidebar.warning(BANNER)
    return df


def main() -> None:
    st.set_page_config(page_title='Irish News AI-likelihood', layout='wide')
    st.title('AI-generated content in Irish news — explorer')
    st.caption(
        'Interim demo · corpus of 44,864 articles, 5,608 per outlet per '
        'period · pre/post ChatGPT (30 Nov 2022)'
    )
