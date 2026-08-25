# ruff: noqa: RUF001  (en dashes are intentional in the length-band labels)
"""One filter state for the whole app, and the rules for applying it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import streamlit as st

from dashboard import theming

if TYPE_CHECKING:
    import pandas as pd

AGGREGATE_OUTLETS: Final = frozenset({'(category)', 'all', ''})

PERIODS: Final = ('both', 'pre', 'post')
VIEWS: Final = (
    'all',
    'non_wire',
    'len_0-150',
    'len_150-300',
    'len_300-600',
    'len_600-inf',
)
VIEW_LABEL: Final = {
    'all': 'all articles',
    'non_wire': 'excluding wire copy',
    'len_0-150': '0–150 words',
    'len_150-300': '150–300 words',
    'len_300-600': '300–600 words',
    'len_600-inf': '600+ words',
}
MAX_MIN_WORDS: Final = 1500


@dataclass(frozen=True)
class Filters:
    """The complete filter state for one script run.

    Attributes:
        outlets (tuple[str, ...]): Selected outlet slugs.
        period (str): ``both``, ``pre`` or ``post``.
        detectors (tuple[str, ...]): Selected detector slugs.
        fpr_target (float): Calibration false-positive target, 0.01 or 0.05.
        view (str): Reported-table view, e.g. ``all`` or ``len_300-600``.
        min_words (int): Explorer-only minimum article length.
        drop_wire (bool): Explorer-only wire-copy exclusion.

    """

    outlets: tuple[str, ...]
    period: str
    detectors: tuple[str, ...]
    fpr_target: float
    view: str
    min_words: int
    drop_wire: bool

    # ---------- derived ----------
    @property
    def suffix(self) -> str:
        """Return the filename suffix for the selected false-positive target.

        Returns:
            str: ``fpr1`` or ``fpr5``.

        """
        return f'fpr{round(self.fpr_target * 100)}'

    @property
    def all_outlets(self) -> bool:
        """Report whether every outlet is selected.

        Returns:
            bool: ``True`` when the outlet filter is not narrowing anything.

        """
        return set(self.outlets) == set(theming.OUTLETS)

    # ---------- application ----------
    def by_outlet(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Keep the selected outlets, plus any aggregate rows.

        Args:
            frame (pd.DataFrame): Any table.

        Returns:
            pd.DataFrame: The table, narrowed if it has an ``outlet`` column.

        """
        if 'outlet' not in frame.columns:
            return frame
        outlet = frame['outlet'].fillna('')
        keep = outlet.isin(self.outlets) | outlet.isin(AGGREGATE_OUTLETS)
        return frame[keep]

    def by_period(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Keep the selected period, if one is selected and the table has periods.

        Args:
            frame (pd.DataFrame): Any table.

        Returns:
            pd.DataFrame: The table, narrowed if applicable.

        """
        if self.period == 'both' or 'period' not in frame.columns:
            return frame
        return frame[frame['period'] == self.period]

    def by_detector(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Keep the selected detectors, if the table is detector-keyed.

        Args:
            frame (pd.DataFrame): Any table.

        Returns:
            pd.DataFrame: The table, narrowed if it has a ``detector`` column.

        """
        if 'detector' not in frame.columns:
            return frame
        return frame[frame['detector'].isin(self.detectors)]

    def by_fpr(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Keep rows written at the selected false-positive target.

        Args:
            frame (pd.DataFrame): Any table.

        Returns:
            pd.DataFrame: The table, narrowed if it has an ``fpr_target`` column.

        """
        if 'fpr_target' not in frame.columns:
            return frame
        return frame[frame['fpr_target'] == self.fpr_target]

    def pipeline(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply every filter that a pre-computed pipeline table can honour.

        Args:
            frame (pd.DataFrame): A table written by the pipeline.

        Returns:
            pd.DataFrame: The narrowed table.

        """
        return self.by_fpr(self.by_detector(self.by_period(self.by_outlet(frame))))

    def corpus(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the full filter set to the scored corpus.

        Args:
            frame (pd.DataFrame): ``corpus_scored.csv``.

        Returns:
            pd.DataFrame: The filtered corpus.

        """
        if frame.empty:
            return frame
        out = frame[frame['outlet'].isin(self.outlets)]
        if self.min_words:
            out = out[out['word_count'] >= self.min_words]
        out = self.by_period(out)
        if self.drop_wire:
            out = out[out['is_wire'] == 0]
        return out

    def ordered_detectors(self) -> tuple[str, ...]:
        """Return the selected detectors in the canonical reporting order.

        Returns:
            tuple[str, ...]: Detector slugs.

        """
        return tuple(d for d in theming.ALL_DETECTORS if d in self.detectors)


def pick_view(frame: pd.DataFrame, wanted: str) -> tuple[pd.DataFrame, str]:
    """Narrow a table to a view, falling back when that view was not written.

    Args:
        frame (pd.DataFrame): A table with a ``view`` column.
        wanted (str): The requested view.

    Returns:
        tuple[pd.DataFrame, str]: The narrowed table, and the view it used.

    """
    if 'view' not in frame.columns:
        return frame, wanted
    available = set(frame['view'].dropna().unique())
    used = wanted if wanted in available else 'all'
    if used not in available:
        return frame, 'all views'
    return frame[frame['view'] == used], used


# ---------- SIDEBAR ----------
def render() -> Filters:
    """Draw the sidebar controls and return the resulting filter state.

    Returns:
        Filters: The filter state for this run.

    """
    sidebar = st.sidebar
    sidebar.subheader('Filters', help='Each group notes which pages it reaches.')

    sidebar.caption('**Everywhere**')
    picked = sidebar.multiselect(
        'Outlets',
        theming.OUTLETS,
        default=list(theming.OUTLETS),
        format_func=lambda slug: theming.OUTLET_LABEL[slug],
        key='f_outlets',
    )
    if not picked:
        picked = list(theming.OUTLETS)
        sidebar.caption(':grey[No outlet selected — showing all four.]')

    period = (
        sidebar.segmented_control('Period', PERIODS, default='both', key='f_period')
        or 'both'
    )

    detectors = sidebar.pills(
        'Detectors',
        theming.ALL_DETECTORS,
        selection_mode='multi',
        default=list(theming.REPORTED_DETECTORS),
        format_func=lambda slug: theming.DETECTOR_LABEL[slug],
        key='f_detectors',
        help='RADAR is scored but excluded from the reported set — AUC 0.4317 '
        "against The Liberal's human anchor.",
    )
    if not detectors:
        detectors = list(theming.REPORTED_DETECTORS)
        sidebar.caption(':grey[No detector selected — showing the reported three.]')

    fpr = sidebar.segmented_control(
        'False-positive target',
        theming.FPR_TARGETS,
        default=0.05,
        format_func=lambda value: f'{value:.0%}',
        key='f_fpr',
        help='Thresholds are fitted on the held-out human anchor at this target. '
        '1% is the stricter bar and yields lower detected rates.',
    )
    fpr = 0.05 if fpr is None else float(fpr)

    sidebar.caption('**Reported tables only**')
    view = sidebar.selectbox(
        'View',
        VIEWS,
        format_func=lambda name: VIEW_LABEL[name],
        key='f_view',
        help='Reported tables are written per view. A view a table does not carry '
        'falls back to "all articles", and the panel says so.',
    )

    sidebar.caption('**Corpus explorer only**')
    min_words = sidebar.slider(
        'Minimum word count', 0, MAX_MIN_WORDS, 0, 50, key='f_min_words'
    )
    drop_wire = sidebar.checkbox('Exclude wire copy', key='f_drop_wire')

    return Filters(
        outlets=tuple(picked),
        period=period,
        detectors=tuple(detectors),
        fpr_target=fpr,
        view=view,
        min_words=min_words,
        drop_wire=drop_wire,
    )


def caption(filters: Filters, *, reach: str) -> None:
    """Print the one-line summary of what is currently narrowing a page.

    Args:
        filters (Filters): The active filter state.
        reach (str): ``corpus`` for the explorer pages, ``pipeline`` for the reported
            pages, which cannot honour the length and wire controls.

    """
    parts = [
        'outlets: '
        + (
            'all four'
            if filters.all_outlets
            else ', '.join(theming.OUTLET_LABEL[o] for o in filters.outlets)
        ),
        f'period: {filters.period}',
        'detectors: '
        + ', '.join(theming.DETECTOR_LABEL[d] for d in filters.ordered_detectors()),
    ]
    if reach == 'corpus':
        parts.append(f'min words: {filters.min_words}')
        if filters.drop_wire:
            parts.append('wire copy excluded')
    else:
        parts.append(f'FPR target: {filters.fpr_target:.0%}')
        parts.append(f'view: {VIEW_LABEL[filters.view]}')
    st.caption(' · '.join(parts))
