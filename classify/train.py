"""Fine-tune DeBERTa-v3-base to separate human from AI-generated news text."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# ---------- FILE SYSTEM ----------
ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT / 'data' / 'classify'
MODEL_DIR = ROOT / 'models' / 'deberta-v3-base-ai-detect'

# ---------- MODEL ----------
BASE_MODEL = 'microsoft/deberta-v3-base'
LABEL_NAMES = {0: 'human', 1: 'ai'}
AI_LABEL = 1


def _load_split(name: str, tokenizer: object) -> object:
    """Return one tokenised split.

    Args:
        name (str): Split name, matching ``<name>.csv`` in the split directory.
        tokenizer (object): The model's tokenizer.

    Returns:
        object: A tokenised ``datasets.Dataset``.

    Raises:
        FileNotFoundError: If the split has not been built.

    """
    path = SPLIT_DIR / f'{name}.csv'
    if not path.exists():
        message = f'{path.name} not found. Run `python -m classify.dataset` first.'
        raise FileNotFoundError(message)

    frame = pd.read_csv(path)[['text', 'label']]
    dataset = Dataset.from_pandas(frame, preserve_index=False)
    logger.info(
        '%-11s %5d examples (%d human / %d AI)',
        name,
        len(frame),
        int((frame['label'] == 0).sum()),
        int((frame['label'] == AI_LABEL).sum()),
    )
    return dataset.map(
        lambda batch: tokenizer(batch['text'], truncation=True, max_length=MAX_TOKENS),
        batched=True,
        remove_columns=['text'],
    )


def train() -> dict[str, float]:
    """Fine-tune the classifier on CPU and save the best checkpoint.

    Returns:
        dict[str, float]: Validation metrics for the selected checkpoint.

    """
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    return {}


def main() -> int:
    """Fine-tune the classifier.

    Returns:
        int: 0 on success, 1 if the splits are missing.

    """
    try:
        return 1
    except FileNotFoundError:
        logger.exception('missing input')
        return 1
    return 0


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )
    sys.exit(main())
