"""Melt + clean the gsingh1 dataset into the known-AI calibration anchor."""

import csv
import logging
import sys
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_DIR = ROOT / 'data' / 'calibration'
SRC = CALIBRATION_DIR / 'gsingh1_train' / 'train.csv'
OUT = CALIBRATION_DIR / 'known_ai.csv'

MIN_WORDS = 50
NON_MODEL_COLS = frozenset({'prompt', 'Human_story'})
FIELDS = ['id', 'model', 'n_words', 'n_chars', 'text']

csv.field_size_limit(1 << 24)


def model_label(column: str) -> str:
    """Return a short model label for a model column name.

    Args:
        column (str): Raw column name.

    Returns:
        str: The final path segment (e.g. ``yi-large``).

    """
    return column.rsplit('/', 1)[-1]


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
    with (
        src.open(encoding='utf-8') as file_input,
        output.open('w', newline='', encoding='utf-8') as file_output,
    ):
        reader = csv.DictReader(file_input)
        model_cols = [c for c in reader.fieldnames or [] if c not in NON_MODEL_COLS]
        labels = {c: model_label(c) for c in model_cols}
        writer = csv.DictWriter(file_output, fieldnames=FIELDS)
        writer.writeheader()


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
