"""Per-outlet calibration, corpus scoring, and output tables for the detectors."""

from __future__ import annotations

import sys
from pathlib import Path
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# ---------- CONFIG ----------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
CALIBRATION_DIR = DATA / 'calibration'
DETECTION_DIR = DATA / 'detection'

HUMAN_PARSED = CALIBRATION_DIR / 'human_parsed.csv'
INPUTS = DETECTION_DIR / 'score_inputs.csv'

MIN_HUMAN_CHARS = 400


def _human_usable_row(row: pd.Series, seen: set[str]) -> bool:  # noqa: PLR0911
    """Return True if a human-anchor row survives the corpus drop rules.

    The human anchor and the corpus pass identical filters
    from :func:`postprocess.curate.drop_reason`.

    Args:
        row (pd.Series): One row of human_parsed.csv.
        seen (set[str]): Raw body_sha1 hashes already kept; mutated in place.

    Returns:
        bool: True if the row is a usable anchor article.

    """
    if row.get('http_status') != '200':
        return False
    if int(row.get('body_len_raw') or 0) < MIN_HUMAN_CHARS:
        return False
    if not row.get('body_text', '').strip():
        return False
    outlet = row.get('outlet', '')
    if outlet == 'irish_examiner' and row.get('sub_excl') == '1':
        return False
    if outlet == 'gript' and row.get('gript_premium') == '1':
        return False
    if outlet == 'gript' and row.get('is_otd') == '1':
        return False
    body_hash = row.get('body_sha1', '')
    if body_hash:
        if body_hash in seen:
            return False
        seen.add(body_hash)
    return True


def build_inputs() -> Path:
    """Assemble the id -> text table to score (human + AI + corpus).

    Returns:
        Path: The written ``score_inputs.csv``.

    """
    DETECTION_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    if HUMAN_PARSED.exists():
        return 1
    return INPUTS


def main() -> int:
    """Assemble inputs (if needed) and, once scores exist, emit all outputs."""
    if not INPUTS.exists():
        return 0
    return 1
