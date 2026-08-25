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
