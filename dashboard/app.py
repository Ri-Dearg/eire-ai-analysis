"""Streamlit dashboard: AI-likelihood over the Irish-news corpus (local-only)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ---------- DIRECTORIES ----------
ROOT = Path(__file__).resolve().parent.parent
SCORED = ROOT / 'data' / 'corpus_scored.csv'

# ---------- LABEL SETTINGS ----------
OUTLETS = ['rte', 'irish_examiner', 'the_liberal', 'gript']
PALETTE = {
    'rte': '#449fab',
    'irish_examiner': '#f05a22',
    'the_liberal': '#1d72bd',
    'gript': '#df4949',
}
BANNER = (
    'Detector scores are a **lower bound**, not a count. '
    'Pre-ChatGPT articles are a false-positive control. '
    'All processing is local; no article text is loaded.'
)

DETECTORS = {
    'radar_score': 'RADAR (P(AI))',
    'perplexity_score': 'Perplexity baseline (uncalibrated)',
    'fastdetectgpt_score': 'Fast-DetectGPT (partial)',
    'binoculars_score': 'Binoculars (partial)',
}


# ---------- DATASET ----------
@st.cache_data
def load() -> pd.DataFrame:
    """Load the freshest scored corpus CSV (metadata + scores, no bodies).

    Returns:
        pd.DataFrame: DataFrame to Display.

    """
    src = SCORED
    df = pd.read_csv(src)
    df['word_count'] = pd.to_numeric(df['word_count'], errors='coerce')
    df['is_wire'] = pd.to_numeric(df['is_wire'], errors='coerce').fillna(0)
    return df


# ---------- HELPER ----------
# AI Designed
def _grouped_box(df: pd.DataFrame, col: str, title: str, ylabel: str) -> plt.Figure:
    """Box plot of col by outlet x period.

    Args:
        df (pd.DataFrame): DataFrame to display
        col (str): Column to show
        title (str): Title for Plot
        ylabel (str): Label for Y

    Returns:
        plt.Figure: Plot to display.

    """
    fig, ax = plt.subplots(figsize=(10, 4.5))
    positions, data, colours, ticks = [], [], [], []
    pos = 0
    present = [outlet for outlet in OUTLETS if outlet in set(df['outlet'])]
    for outlet in present:
        for period in ('pre', 'post'):
            sub = df[(df['outlet'] == outlet) & (df['period'] == period)]
            data.append(sub[col].dropna())
            positions.append(pos)
            colours.append(PALETTE[outlet] if period == 'post' else '#cccccc')
            ticks.append((pos, period.upper()))
            pos += 1
        ax.text(
            pos - 1.5,
            -0.22,
            outlet,
            ha='center',
            fontsize=11,
            transform=ax.get_xaxis_transform(),
        )
        pos += 0.6
    if not data:
        return fig
    bp = ax.boxplot(
        data,
        positions=positions,
        patch_artist=True,
        showfliers=False,
        widths=0.7,
        medianprops={'color': 'black', 'linewidth': 1.6},
    )
    for patch, colour in zip(bp['boxes'], colours, strict=True):
        patch.set_facecolor(colour)
        patch.set_alpha(0.85)
    ax.set_xticks([t for t, _ in ticks])
    ax.set_xticklabels([lab for _, lab in ticks], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    return fig


def apply_filters(
    df: pd.DataFrame, outlets: list[str], period: str, min_words: int, *, drop_wire: bool
) -> pd.DataFrame:
    """Return the filtered frame shared by every tab.

    Args:
        df (pd.DataFrame): Dataframe to filter.
        outlets (list[str]): Outlets to filter.
        period (str): Pre or Post ChatGPT period.
        min_words (int): Minimum amount of words per article.
        drop_wire (bool): Remove wire articles.

    Returns:
        pd.DataFrame: Filtered Dataframe.

    """
    output = df[df['outlet'].isin(outlets) & (df['word_count'] >= min_words)]
    if period != 'both':
        output = output[output['period'] == period]
    if drop_wire:
        output = output[output['is_wire'] == 0]
    return output


# ---------- LAYOUT ----------
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
    return apply_filters(df, picked, period, min_words, drop_wire=drop_wire)


def tab_articles(df: pd.DataFrame) -> None:
    """Article-level drill-down, metadata and scores only.

    Args:
        df (pd.DataFrame): DataFrame to display.

    """
    cols = [
        'article_id',
        'outlet',
        'published_date',
        'section',
        'author',
        'word_count',
        'is_wire',
        *DETECTORS,
    ]
    show = df[cols].copy()
    st.dataframe(show, width='stretch', hide_index=True, height=520)
    st.caption(
        'No article text is stored in this app — metadata and scores '
        'only (local/legal constraint).'
    )


def tab_comparison(df: pd.DataFrame) -> None:
    """Legacy vs counter-consensus summary.

    Args:
        df (pd.DataFrame): DataFrame to display.

    """
    st.info(
        'Legacy vs Counter-consensus. **Pending full ensemble scoring and calibration**'
    )
    med = df.groupby(['category', 'period'])[list(DETECTORS)].median().round(4)
    st.dataframe(med, width='stretch')
    st.caption(
        'Medians of raw detector scores under the current filters. '
        'Raw scores are not comparable across detectors and are not '
        'calibrated probabilities.'
    )


def tab_detectors(df: pd.DataFrame) -> None:
    """Score distributions for one detector.

    Args:
        df (pd.DataFrame): DataFrame to display.

    """
    col = st.selectbox('Detector', options=list(DETECTORS), format_func=DETECTORS.get)
    scored = df[df[col].notna()]
    if col == 'radar_score':
        st.info(
            'Raise the **minimum word count** slider (sidebar) to ~300 and '
            'watch The Liberal PRE box collapse: the spike is a short-text '
            'false positive, not pre-2022 AI.'
        )
    st.caption(
        f'{len(scored):,} of {len(df):,} filtered articles carry a '
        f'{DETECTORS[col]} score.'
    )
    st.pyplot(_grouped_box(scored, col, DETECTORS[col], 'score'), width='stretch')


def tab_overview(df: pd.DataFrame) -> None:
    """Corpus composition and the length picture.

    Args:
        df (pd.DataFrame): DataFrame to display

    """
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Articles (filtered)', f'{len(df):,}')
    col2.metric('Outlets', df['outlet'].nunique())
    col3.metric('Median words', f'{df["word_count"].median():,.0f}' if len(df) else '—')
    col4.metric('Wire share', f'{df["is_wire"].mean():.1%}' if len(df) else '—')
    comp = (
        df.pivot_table(index='outlet', columns='period', aggfunc='size', fill_value=0)
        .reindex(OUTLETS)
        .dropna(how='all')
    )
    st.subheader('Composition (articles per outlet × period)')
    st.bar_chart(comp)
    st.subheader('Article length by outlet × period')
    st.pyplot(_grouped_box(df, 'word_count', '', 'words per article'), width='stretch')


def main() -> None:
    """Run dashboard."""
    st.set_page_config(page_title='Irish News AI-likelihood', layout='wide')
    st.title('AI-generated content in Irish news — explorer')
    st.caption(
        'Interim demo · corpus of 44,864 articles, 5,608 per outlet per '
        'period · pre/post ChatGPT (30 Nov 2022)'
    )
    df = sidebar(load())

    t_over, t_det, t_cmp, t_art = st.tabs(
        ['Overview', 'Detectors', 'Comparison', 'Articles']
    )
    with t_over:
        tab_overview(df)
    with t_det:
        tab_detectors(df)
    with t_cmp:
        tab_comparison(df)
    with t_art:
        tab_articles(df)


if __name__ == '__main__':
    main()
