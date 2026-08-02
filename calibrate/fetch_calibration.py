"""Fetch the human calibration articles into a separate, isolated database."""

from __future__ import annotations

import sys
from pathlib import Path

# ---------- CONFIG ----------
ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_DIR = ROOT / 'data' / 'calibration'
CAL_DB = str(CALIBRATION_DIR / 'calibration.db')
SOURCE_FEED = 'calibration'
LEGACY = ('rte', 'irish_examiner', 'the_liberal')


def main() -> int:
    """Ingest every outlet's calibration URLs into the isolated calibration DB."""
    if not Path(CAL_DB).exists():
        print(
            f'ERROR: {CAL_DB} not found. Create it first (schema + outlets) -- '
            'see RUNBOOK.md.',
            file=sys.stderr,
        )
        return 1
    return 0
