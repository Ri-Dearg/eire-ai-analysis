# ruff: noqa: RUF001, RUF002  (en dashes and multiplication signs are in labels)
"""Streamlit dashboard: AI-likelihood over the Irish-news corpus (local-only)."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pandas as pd
import streamlit as st

# `streamlit run dashboard/app.py` puts dashboard/ on sys.path, not the repo root, so
# the package import below would fail. Both entry points -- that one and
# `python -m dashboard` -- work once the root is on the path.
if __package__ in {None, ''}:  # pragma: no cover - import-path bootstrap
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard import divergence, results, theming
from dashboard import filters as filters_module

if TYPE_CHECKING:
    from dashboard.filters import Filters

# ---------- DIRECTORIES ----------
ROOT = Path(__file__).resolve().parent.parent
SCORED = ROOT / 'data' / 'corpus_scored.csv'

CORPUS_ROWS: Final = 44_864
PER_CELL: Final = 5_608
CHATGPT_RELEASE: Final = '30 Nov 2022'

BANNER: Final = (
    'Detector scores are a **lower bound**, not a count. Pre-ChatGPT articles are a '
    'false-positive control. All processing is local; no article text is loaded.'
)

DETECTOR_COLUMN: Final = {
    detector: f'{detector}_score' for detector in theming.ALL_DETECTORS
}

REQUIRED_COLUMNS: Final = (
    'article_id',
    'outlet',
    'published_date',
    'period',
    'category',
    'section',
    'author',
    'word_count',
    'is_wire',
    *DETECTOR_COLUMN.values(),
)


# ---------- DATASET ----------
@st.cache_data(show_spinner='Loading scored corpus…')
def load() -> pd.DataFrame:
    """Load the scored corpus CSV (metadata and scores, no article bodies).

    Returns:
        pd.DataFrame: The scored corpus, or an empty frame with the expected columns
        when the file is absent.

    """
    if not SCORED.exists():
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS))
    frame = pd.read_csv(SCORED)
    frame['word_count'] = pd.to_numeric(frame['word_count'], errors='coerce')
    frame['is_wire'] = pd.to_numeric(frame['is_wire'], errors='coerce').fillna(0)
    return frame


def _score_columns(filters: Filters) -> list[str]:
    """Return the score columns for the selected detectors that the corpus carries.

    Args:
        filters (Filters): The active filter state.

    Returns:
        list[str]: Column names.

    """
    return [DETECTOR_COLUMN[detector] for detector in filters.ordered_detectors()]


def _grouped_box(
    frame: pd.DataFrame, column: str, title: str, ylabel: str, filters: Filters
) -> None:
    """Draw a box plot of one column by outlet × period.

    Args:
        frame (pd.DataFrame): The filtered corpus.
        column (str): Column to plot.
        title (str): Plot title.
        ylabel (str): Y-axis label.
        filters (Filters): The active filter state, for outlet order and colour.

    """
    periods = ('pre', 'post') if filters.period == 'both' else (filters.period,)
    present = [outlet for outlet in filters.outlets if outlet in set(frame['outlet'])]
    if not present:
        st.info(
            'No articles match the current filters.', icon=':material/filter_alt_off:'
        )
        return

    figure, axis = theming.figure(width=10, height=4.5)
    colours = theming.palette()
    shade = theming.ink()
    positions, data, patches, ticks = [], [], [], []
    position = 0.0
    for outlet in present:
        for period in periods:
            cell = frame[(frame['outlet'] == outlet) & (frame['period'] == period)]
            data.append(cell[column].dropna())
            positions.append(position)
            patches.append(colours[outlet] if period == 'post' else shade['neutral'])
            ticks.append((position, period.upper()))
            position += 1
        axis.text(
            position - (len(periods) + 1) / 2,
            -0.22,
            theming.OUTLET_LABEL[outlet],
            ha='center',
            fontsize=10,
            transform=axis.get_xaxis_transform(),
        )
        position += 0.6

    if not any(len(series) for series in data):
        st.info(
            'No scored articles match the current filters.',
            icon=':material/filter_alt_off:',
        )
        return
    plot = axis.boxplot(
        data,
        positions=positions,
        patch_artist=True,
        showfliers=False,
        widths=0.7,
        medianprops={'color': shade['text'], 'linewidth': 1.6},
        whiskerprops={'color': shade['axis']},
        capprops={'color': shade['axis']},
        boxprops={'color': shade['axis']},
    )
    for patch, colour in zip(plot['boxes'], patches, strict=True):
        patch.set_facecolor(colour)
        patch.set_alpha(0.85)
    axis.set_xticks([tick for tick, _ in ticks])
    axis.set_xticklabels([label for _, label in ticks], fontsize=9)
    axis.set_ylabel(ylabel)
    if title:
        axis.set_title(title)
    theming.show(figure)


# ---------- EXPLORER PAGES ----------
def page_overview(frame: pd.DataFrame, filters: Filters) -> None:
    """Show corpus composition and the length picture.

    Args:
        frame (pd.DataFrame): The filtered corpus.
        filters (Filters): The active filter state.

    """
    st.subheader('Corpus composition')
    columns = st.columns(4)
    columns[0].metric('Articles (filtered)', f'{len(frame):,}')
    columns[1].metric('Outlets', frame['outlet'].nunique())
    columns[2].metric(
        'Median words', f'{frame["word_count"].median():,.0f}' if len(frame) else '—'
    )
    columns[3].metric(
        'Wire share', f'{frame["is_wire"].mean():.1%}' if len(frame) else '—'
    )
    if frame.empty:
        st.info(
            'No articles match the current filters.', icon=':material/filter_alt_off:'
        )
        return

    composition = (
        frame.pivot_table(index='outlet', columns='period', aggfunc='size', fill_value=0)
        .reindex([o for o in theming.OUTLETS if o in set(frame['outlet'])])
        .dropna(how='all')
    )
    composition.index = theming.label_outlets(composition.index)
    st.bar_chart(composition, height=260)
    st.caption(
        f'The frozen corpus is {CORPUS_ROWS:,} articles, {PER_CELL:,} per outlet per '
        f'period, split at the ChatGPT release ({CHATGPT_RELEASE}). Any imbalance above '
        'is the filters, not the corpus.'
    )
    st.divider()
    st.subheader('Article length by outlet × period')
    _grouped_box(frame, 'word_count', '', 'words per article', filters)


def page_detectors(frame: pd.DataFrame, filters: Filters) -> None:
    """Show raw score distributions for one detector.

    Args:
        frame (pd.DataFrame): The filtered corpus.
        filters (Filters): The active filter state.

    """
    st.subheader('Raw detector score distributions')
    detector = st.selectbox(
        'Detector',
        filters.ordered_detectors(),
        format_func=theming.DETECTOR_LABEL.get,
        key='explorer_detector',
    )
    column = DETECTOR_COLUMN[detector]
    if column not in frame.columns:
        st.info('The scored corpus does not carry that column.')
        return
    scored = frame[frame[column].notna()]
    if detector == 'radar':
        st.warning(
            'RADAR is shown for completeness and is **excluded from the reported '
            "ensemble** — AUC 0.4317 against The Liberal's human anchor, i.e. worse "
            'than chance on the outlet that matters most. Raise the **minimum word '
            'count** to ~300 and watch The Liberal PRE box collapse: the spike is a '
            'short-text false positive, not pre-2022 AI.'
        )
    st.caption(
        f'{len(scored):,} of {len(frame):,} filtered articles carry a '
        f'{theming.DETECTOR_LABEL[detector]} score.'
    )
    _grouped_box(scored, column, theming.DETECTOR_LABEL[detector], 'score', filters)
    st.caption(
        'Raw scores, higher = more AI-like. They are **not** calibrated probabilities '
        'and are not comparable across detectors — the Calibration page is where they '
        'become comparable.'
    )


def page_comparison(frame: pd.DataFrame, filters: Filters) -> None:
    """Show the legacy vs counter-consensus summary over raw scores.

    Args:
        frame (pd.DataFrame): The filtered corpus.
        filters (Filters): The active filter state.

    """
    st.subheader('Legacy vs counter-consensus')
    columns = _score_columns(filters)
    if frame.empty or not columns:
        st.info(
            'No articles match the current filters.', icon=':material/filter_alt_off:'
        )
        return
    medians = frame.groupby(['category', 'period'])[columns].median().round(4)
    medians.columns = [
        theming.DETECTOR_LABEL[name.removesuffix('_score')] for name in medians.columns
    ]
    st.dataframe(medians, width='stretch')
    counts = frame.groupby(['category', 'period']).size().rename('articles')
    st.dataframe(counts.to_frame().T, width='stretch')
    st.caption(
        'Medians of **raw** detector scores under the current filters — a descriptive '
        'summary, not the test. The pre-specified comparison is a Mann-Whitney U with '
        "Cliff's δ and lives on the **Effect sizes** page; a difference in medians here "
        'carries no significance claim.'
    )


def page_articles(frame: pd.DataFrame, filters: Filters) -> None:
    """Show the article-level drill-down: metadata and scores only.

    Args:
        frame (pd.DataFrame): The filtered corpus.
        filters (Filters): The active filter state.

    """
    st.subheader('Articles')
    columns = [
        'article_id',
        'outlet',
        'published_date',
        'section',
        'author',
        'word_count',
        'is_wire',
        *_score_columns(filters),
    ]
    present = [name for name in columns if name in frame.columns]
    show = frame[present].copy()
    if 'outlet' in show.columns:
        show['outlet'] = show['outlet'].map(theming.OUTLET_LABEL).fillna(show['outlet'])
    st.dataframe(
        show,
        width='stretch',
        hide_index=True,
        height=520,
        column_config={
            'article_id': st.column_config.NumberColumn('id', format='%d'),
            'word_count': st.column_config.NumberColumn('words', format='%d'),
            'is_wire': st.column_config.CheckboxColumn('wire'),
            **{
                DETECTOR_COLUMN[detector]: st.column_config.NumberColumn(
                    theming.DETECTOR_LABEL[detector], format='%.3f'
                )
                for detector in filters.ordered_detectors()
            },
        },
    )
    st.caption(
        f'{len(show):,} rows. No article text is stored in this app — metadata and '
        'scores only, which is the constraint the corpus was collected under.'
    )


# ---------- FRAME ----------
PageSpec = tuple[str, Callable[..., None], str, str, str, bool]

PAGE_TABLE: Final[tuple[PageSpec, ...]] = (
    ('Corpus', page_overview, 'Overview', ':material/dataset:', 'overview', True),
    (
        'Corpus',
        page_detectors,
        'Detector scores',
        ':material/insights:',
        'detector-scores',
        True,
    ),
    (
        'Corpus',
        results.page_time_series,
        'Over time',
        ':material/timeline:',
        'over-time',
        True,
    ),
    ('Corpus', page_articles, 'Articles', ':material/table_rows:', 'articles', True),
    (
        'Findings',
        results.page_results,
        'Effect sizes',
        ':material/compare_arrows:',
        'effect-sizes',
        False,
    ),
    (
        'Findings',
        divergence.page_divergence,
        'Divergence',
        ':material/trending_up:',
        'divergence',
        False,
    ),
    (
        'Findings',
        page_comparison,
        'Raw comparison',
        ':material/balance:',
        'raw-comparison',
        True,
    ),
    (
        'Method',
        results.page_calibration,
        'Calibration',
        ':material/tune:',
        'calibration',
        False,
    ),
    (
        'Method',
        results.page_prevalence,
        'Prevalence',
        ':material/percent:',
        'prevalence',
        False,
    ),
    (
        'Method',
        results.page_generation,
        'Generation ladder',
        ':material/science:',
        'generation',
        False,
    ),
    (
        'Method',
        results.page_classifier,
        'Classifier',
        ':material/model_training:',
        'classifier',
        False,
    ),
)


def _bind(page: Callable[..., None], *, needs_corpus: bool) -> Callable[[], None]:
    """Wrap a page function so ``st.navigation`` can call it with no arguments.

    The filter state and the filtered corpus are read from session state, which is
    per-session; module-level state would be shared across every viewer of a deployed
    app.

    Args:
        page (Callable[..., None]): The page function.
        needs_corpus (bool): Whether the page takes the filtered corpus as its first
            argument.

    Returns:
        Callable[[], None]: A zero-argument callable.

    """

    def run() -> None:
        active = st.session_state['filters']
        if needs_corpus:
            page(st.session_state['corpus'], active)
        else:
            page(active)

    run.__name__ = page.__name__
    return run


def _pages() -> tuple[dict[str, list[st.Page]], dict[str, bool]]:
    """Build the navigation tree and the per-page filter-reach lookup.

    Returns:
        tuple[dict[str, list[st.Page]], dict[str, bool]]: Section label to pages, and
        page title to whether that page recomputes from the corpus.

    """
    tree: dict[str, list[st.Page]] = {}
    reach: dict[str, bool] = {}
    for index, (section, function, title, icon, url, corpus) in enumerate(PAGE_TABLE):
        tree.setdefault(section, []).append(
            st.Page(
                _bind(function, needs_corpus=corpus),
                title=title,
                icon=icon,
                url_path=url,
                default=index == 0,
            )
        )
        reach[title] = corpus
    return tree, reach


def main() -> None:
    """Run the dashboard: frame, sidebar and routing."""
    st.set_page_config(
        page_title='Irish news AI-likelihood',
        page_icon=':material/newspaper:',
        layout='wide',
    )
    theming.apply_matplotlib()

    st.title('AI-generated content in Irish news')
    st.caption(
        f'Evidence explorer · {CORPUS_ROWS:,} articles, {PER_CELL:,} per outlet per '
        f'period · pre/post ChatGPT ({CHATGPT_RELEASE}) · RTÉ and the Irish Examiner '
        '(legacy) against The Liberal and Gript (counter-consensus)'
    )

    tree, reach = _pages()
    page = st.navigation(tree)

    raw = load()
    if raw.empty:
        st.warning(
            'No scored corpus on disk. Every panel will show what it would display '
            'and the command that produces it. Run `python -m detect` to build '
            '`data/corpus_scored.csv`.',
            icon=':material/database_off:',
        )
    active = filters_module.render()
    st.session_state['filters'] = active
    st.session_state['corpus'] = active.corpus(raw)

    st.sidebar.divider()
    st.sidebar.warning(BANNER, icon=':material/warning:')

    filters_module.caption(
        active, reach='corpus' if reach.get(page.title, False) else 'pipeline'
    )
    page.run()


if __name__ == '__main__':
    main()
