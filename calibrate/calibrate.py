"""Per-outlet calibration, corpus scoring, and output tables for the detectors."""

from __future__ import annotations

import sys
from pathlib import Path

# ---------- CONFIG ----------

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
CALIBRATION_DIR = DATA / 'calibration'
DETECTION_DIR = DATA / 'detection'

INPUTS = DETECTION_DIR / 'score_inputs.csv'


def main() -> int:
    """Assemble inputs (if needed) and, once scores exist, emit all outputs."""
    if not INPUTS.exists():
        return 0
    return 1
