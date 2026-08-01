"""Melt + clean the gsingh1 dataset into the known-AI calibration anchor."""

import csv
import logging
import re
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

# Refusal regex structured by AI.
# Refusal / non-article model output (checked near the article start). The
# apostrophe class matches the straight quote and the curly U+2019 the dataset
# uses (escaped so no ambiguous glyph appears in source).
REFUSAL_RE = re.compile(
    r"i['’]?m sorry|i am sorry|i cannot|i can['’]?t\b|i can not|"
    r"as an ai\b|as a language model|i do(?:n['’]?t| not) have access|"
    r"i can['’]?t access|cannot access|my training data|"
    r'as of my last (?:update|knowledge)|knowledge cutoff|'
    r"do(?:n['’]?t| not) have real-?time|can['’]?t provide real-?time|"
    r"unable to provide real-?time|i['’]?m unable to|i am unable to",
    re.IGNORECASE,
)

# API / transport error strings that leaked into cells (checked anywhere).
ERROR_RE = re.compile(
    r'Error communicating with OpenAI|Max retries exceeded|getaddrinfo failed|'
    r'HTTPSConnectionPool|NameResolutionError',
    re.IGNORECASE,
)


def is_refusal_or_error(text: str) -> str | None:
    """Return a drop reason if text is a refusal or error, else None.

    Args:
        text (str): Raw model cell.

    Returns:
        str | None: Reason or None

    """
    if ERROR_RE.search(text):
        return 'api_error'
    if REFUSAL_RE.search(text[:500]):
        return 'refusal'
    return None


def model_label(column: str) -> str:
    """Return a short model label for a model column name.

    Args:
        column (str): Raw column name.

    Returns:
        str: The final path segment (e.g. ``yi-large``).

    """
    return column.rsplit('/', 1)[-1]


def clean_cell(raw: str, seen: set[str]) -> tuple[str | None, str]:
    """Clean one model cell; return (text_or_None, reason).

    Args:
        raw (str): Raw model cell.
        seen (set[str]): Normalised-text hashes already kept.

    Returns:
        tuple[str | None, str]: Cleaned text (or None) and reason/''.

    """
    if not raw or not raw.strip():
        return None, 'empty'
    reason = is_refusal_or_error(raw)

    return ''


def clean(src: Path, output: Path):
    """Melt + clean the dataset, streaming rows to out (checkpoint-resumable).

    Args:
        src (Path): Path to train.csv.
        output (Path): Output known_ai.csv (appended to if resuming).

    Returns:
        tuple[Counter, Counter, int]: Kept-per-model, drop reasons (this run),
            and the number of source rows processed this run.

    """
    seen: set[str] = set()
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
        for i, row in enumerate(reader):
            for col in model_cols:
                text, reason = clean_cell(row[col] or '', seen)


def main() -> int:
    """Clean the gsingh1 dataset into known_ai.csv (resumable) + summarise."""
    if not SRC.exists():
        logger.error(
            'ERROR: %s not found, please download gsingh1-py/train first (see RUNBOOK.md).',
            SRC,
        )
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
