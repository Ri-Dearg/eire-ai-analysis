"""Score the human calibration anchor (and RADAR's known-AI remainder) separately."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from calibrate.calibrate import (
    DETECTION_DIR as DET_DIR,
)
from calibrate.calibrate import (
    HUMAN_PARSED,
    KNOWN_AI,
)
from calibrate.calibrate import (
    human_anchor_df as human_anchor_frame,
)

HEAVY = ('fastdetectgpt', 'binoculars')


def human_frame() -> pd.DataFrame:
    """Return the usable human-anchor rows as an (id, text) frame.

    Returns:
        pd.DataFrame: Columns ``id`` and ``text``.

    """
    ok = human_anchor_frame()
    return pd.DataFrame(
        {
            'id': 'human:' + ok['outlet'] + ':' + ok['article_id'].astype(str),
            'text': ok['body_text'],
        }
    )


def ai_frame() -> pd.DataFrame:
    """Return the known-AI rows as an (id, text) frame (``ai:<model>:<n>`` ids)."""
    ai = pd.read_csv(KNOWN_AI, dtype=str).fillna('')
    return pd.DataFrame({'id': 'ai:' + ai['id'].astype(str), 'text': ai['text']})


def _pending(frame: pd.DataFrame, detector: str) -> int:
    """Return how many of ``frame``'s ids are not yet in a detector checkpoint."""
    path = DET_DIR / f'{detector}.csv'
    if not path.exists():
        return len(frame)
    done = set(pd.read_csv(path, usecols=[0]).iloc[:, 0])
    return int((~frame['id'].isin(done)).sum())
