"""Shared vocabulary and theme plumbing for every dashboard panel."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import matplotlib as mpl
import matplotlib.pyplot as plt
import streamlit as st

if TYPE_CHECKING:
    from collections.abc import Iterator

# ---------- VOCABULARY ----------
OUTLETS: Final = ('rte', 'irish_examiner', 'the_liberal', 'gript')
OUTLET_LABEL: Final = {
    'rte': 'RTÉ',
    'irish_examiner': 'Irish Examiner',
    'the_liberal': 'The Liberal',
    'gript': 'Gript',
}
CATEGORY: Final = {
    'rte': 'legacy',
    'irish_examiner': 'legacy',
    'the_liberal': 'counter-consensus',
    'gript': 'counter-consensus',
}

# Report-figure hues (light) and their lightened dark-ground counterparts.
PALETTE_LIGHT: Final = {
    'rte': '#449fab',
    'irish_examiner': '#f05a22',
    'the_liberal': '#1d72bd',
    'gript': '#df4949',
}
PALETTE_DARK: Final = {
    'rte': '#5cc0cd',
    'irish_examiner': '#ff7c47',
    'the_liberal': '#4a9de0',
    'gript': '#f26a6a',
}

REPORTED_DETECTORS: Final = ('fastdetectgpt', 'binoculars', 'perplexity')
ALL_DETECTORS: Final = (*REPORTED_DETECTORS, 'radar')
DETECTOR_LABEL: Final = {
    'fastdetectgpt': 'Fast-DetectGPT',
    'binoculars': 'Binoculars',
    'perplexity': 'Perplexity',
    'radar': 'RADAR',
}
DETECTOR_COLOUR_LIGHT: Final = {
    'fastdetectgpt': '#2c3e50',
    'binoculars': '#8e44ad',
    'perplexity': '#16a085',
    'radar': '#95a5a6',
}
DETECTOR_COLOUR_DARK: Final = {
    'fastdetectgpt': '#7fa8c9',
    'binoculars': '#b98ad6',
    'perplexity': '#3fbf9c',
    'radar': '#8b95a3',
}

MATERIALITY: Final = 0.15
FPR_TARGETS: Final = (0.01, 0.05)

_INK_DARK: Final = {
    'text': '#e6edf3',
    'muted': '#9aa7b8',
    'grid': '#2b323f',
    'axis': '#3a4353',
    'accent': '#4aa3df',
    'warn': '#e06c6c',
    'neutral': '#6b7789',
}
_INK_LIGHT: Final = {
    'text': '#1c2833',
    'muted': '#5b6b7d',
    'grid': '#e2e6eb',
    'axis': '#98a2ad',
    'accent': '#1d72bd',
    'warn': '#c0392b',
    'neutral': '#bdc3c7',
}


# ---------- THEME RESOLUTION ----------
def is_dark() -> bool:
    """Report whether the viewer is currently on the dark theme.

    Returns:
        bool: ``True`` when the active theme is dark.

    """
    try:
        return st.context.theme.type != 'light'
    except (AttributeError, RuntimeError):
        return True


def palette() -> dict[str, str]:
    """Return the outlet colour map for the active theme.

    Returns:
        dict[str, str]: Outlet slug to hex colour.

    """
    return PALETTE_DARK if is_dark() else PALETTE_LIGHT


def detector_colours() -> dict[str, str]:
    """Return the detector colour map for the active theme.

    Returns:
        dict[str, str]: Detector slug to hex colour.

    """
    return DETECTOR_COLOUR_DARK if is_dark() else DETECTOR_COLOUR_LIGHT


def ink() -> dict[str, str]:
    """Return the semantic chart colours for the active theme.

    Returns:
        dict[str, str]: Role name (``text``, ``muted``, ``grid``, ``axis``,
        ``accent``, ``warn``, ``neutral``) to hex colour.

    """
    return _INK_DARK if is_dark() else _INK_LIGHT


def heatmap_cmap() -> str:
    """Return the sequential colormap that reads on the active ground.

    Returns:
        str: A matplotlib colormap name.

    """
    return 'viridis' if is_dark() else 'YlGnBu'


def apply_matplotlib() -> None:
    """Point matplotlib's rcParams at the active Streamlit theme.

    Figure and axes patches are left transparent so the Streamlit background shows
    through; only the ink is set. Call once per script run, before any panel draws.
    """
    shade = ink()
    mpl.rcParams.update(
        {
            'figure.facecolor': 'none',
            'figure.edgecolor': 'none',
            'savefig.facecolor': 'none',
            'axes.facecolor': 'none',
            'axes.edgecolor': shade['axis'],
            'axes.labelcolor': shade['text'],
            'axes.titlecolor': shade['text'],
            'axes.grid': False,
            'text.color': shade['text'],
            'xtick.color': shade['muted'],
            'ytick.color': shade['muted'],
            'grid.color': shade['grid'],
            'legend.frameon': False,
            'legend.labelcolor': shade['text'],
            'font.size': 10,
        }
    )


def figure(*, width: float = 9.0, height: float = 4.2) -> tuple[plt.Figure, plt.Axes]:
    """Create a figure and axes with the house spine treatment already applied.

    Args:
        width (float): Figure width in inches.
        height (float): Figure height in inches.

    Returns:
        tuple[plt.Figure, plt.Axes]: The new figure and its axes.

    """
    fig, axis = plt.subplots(figsize=(width, height))
    axis.spines[['top', 'right']].set_visible(False)
    return fig, axis


def show(fig: plt.Figure) -> None:
    """Render a figure and close it.

    Args:
        fig (plt.Figure): The figure to draw.

    """
    fig.tight_layout()
    st.pyplot(fig, width='stretch')
    plt.close(fig)


def materiality_guides(axis: plt.Axes, *, vertical: bool = True) -> None:
    """Draw the zero line and the two materiality thresholds on an effect axis.

    Args:
        axis (plt.Axes): Axis to draw on.
        vertical (bool): Whether the effect size is on the x-axis.

    """
    shade = ink()
    line = axis.axvline if vertical else axis.axhline
    line(0, color=shade['muted'], linewidth=1)
    for bound in (-MATERIALITY, MATERIALITY):
        line(bound, color=shade['warn'], linestyle=':', linewidth=1)


def label_outlets(names: Iterator[str] | list[str]) -> list[str]:
    """Map outlet slugs to display names, passing unknown values through.

    Args:
        names (Iterator[str] | list[str]): Outlet slugs.

    Returns:
        list[str]: Display names.

    """
    return [OUTLET_LABEL.get(name, name) for name in names]
