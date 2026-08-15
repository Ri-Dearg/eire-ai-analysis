"""Build balanced, leakage-free train/validation/test splits for the classifier."""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ---------- FILE SYSTEM ----------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
CALIBRATION_DIR = DATA / 'calibration'
OUTPUT_DIR = DATA / 'classify'

GSINGH_SRC = CALIBRATION_DIR / 'gsingh1_train' / 'train.csv'
KNOWN_AI = CALIBRATION_DIR / 'known_ai.csv'

HUMAN_COLUMN = 'Human_story'
MIN_WORDS = 50
SEED = 37


LABEL_HUMAN = 0
LABEL_AI = 1
COLUMNS = ('id', 'prompt_id', 'model', 'label', 'n_words', 'text')


def _read_human() -> pd.DataFrame:
    """Return the human stories from the gsingh1 source file.

    Args:
        None.

    Returns:
        pd.DataFrame: One row per usable human story, with its prompt id.

    Raises:
        FileNotFoundError: If the gsingh1 source file is absent.

    """
    if not GSINGH_SRC.exists():
        message = (
            f'{GSINGH_SRC} not found. It is the raw gsingh1-py/train download; '
            'clean_gsingh.py reads the same file.'
        )
        raise FileNotFoundError(message)

    csv.field_size_limit(10**9)
    records = []
    with GSINGH_SRC.open(newline='', encoding='utf-8') as handle:
        for prompt_id, row in enumerate(csv.DictReader(handle)):
            text = (row.get(HUMAN_COLUMN) or '').strip()
            word_count = len(text.split())
            if word_count < MIN_WORDS:
                continue
            records.append(
                {
                    'id': f'human:{prompt_id}',
                    'prompt_id': prompt_id,
                    'model': 'human',
                    'label': LABEL_HUMAN,
                    'n_words': word_count,
                    'text': text,
                }
            )
    logger.info('human stories kept: %d', len(records))
    return pd.DataFrame(records)


def main() -> int:
    """Build the classifier splits.

    Returns:
        int: 0 on success, 1 if a required input is missing.

    """
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except FileNotFoundError:
        logger.exception('missing input')
        return 1
    logger.info('wrote splits to %s', OUTPUT_DIR)
    return 0


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )
    sys.exit(main())
