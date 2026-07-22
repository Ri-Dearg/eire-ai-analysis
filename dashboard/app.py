from __future__ import annotations

from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
SCORED = ROOT / 'data' / 'corpus_scored.csv'


def main() -> None:
    st.set_page_config(page_title='Irish News AI-likelihood', layout='wide')
    st.title('AI-generated content in Irish news — explorer')
    st.caption(
        'Interim demo · corpus of 44,864 articles, 5,608 per outlet per '
        'period · pre/post ChatGPT (30 Nov 2022)'
    )
