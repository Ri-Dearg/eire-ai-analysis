"""Melt + clean the gsingh1 dataset into the known-AI calibration anchor."""

import csv
import hashlib
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_DIR = ROOT / 'data' / 'calibration'
SRC = CALIBRATION_DIR / 'gsingh1_train' / 'train.csv'
OUT = CALIBRATION_DIR / 'known_ai.csv'

MIN_WORDS = 50
NON_MODEL_COLS = frozenset({'prompt', 'Human_story'})

csv.field_size_limit(1 << 24)


def _resume(output: Path) -> tuple[set[str], int]:
    """Return (dedup hashes, last source-row index) from an existing output.

    Lets the pass resume after an interruption.

    Args:
        output (Path): Existing ``known_ai.csv`` (may be absent).

    Returns:
        tuple[set[str], int]: Kept text hashes and the last processed row index
            (-1 if the file is absent/empty).

    """
    seen: set[str] = set()
    last_row = -1
    if not output.exists():
        return seen, last_row
    with output.open(encoding='utf-8') as file:
        for row in csv.DictReader(file):
            seen.add(hashlib.sha1(row['text'].encode('utf-8')).hexdigest())
            last_row = max(last_row, int(row['id'].rsplit(':', 1)[1]))
    return seen, last_row


def clean(src: Path, output: Path):
    """Melt + clean the dataset, streaming rows to ``out`` (checkpoint-resumable).

    Args:
        src (Path): Path to ``train.csv``.
        output (Path): Output ``known_ai.csv`` (appended to if resuming).

    Returns:
        tuple[Counter, Counter, int]: Kept-per-model, drop reasons (this run),
            and the number of source rows processed this run.

    """
    output.parent.mkdir(parents=True, exist_ok=True)
    seen, last = _resume(output)


def main() -> int:
    """Clean the gsingh1 dataset into ``known_ai.csv`` (resumable) + summarise."""
    if not SRC.exists():
        logger.error(
            'ERROR: %s not found, please download gsingh1-py/train first (see RUNBOOK.md).',
            SRC,
        )
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
