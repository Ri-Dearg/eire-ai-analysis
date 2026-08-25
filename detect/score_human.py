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
