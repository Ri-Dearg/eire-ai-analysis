"""Build balanced, leakage-free train/validation/test splits for the classifier."""

from __future__ import annotations

import csv
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from calibrate.clean_gsingh import normalise

logger = logging.getLogger(__name__)


# ---------- FILE SYSTEM ----------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
ANCHOR = DATA / 'calibration' / 'human_parsed.csv'
GENERATED = DATA / 'generation' / 'generated_irish_ai.csv'
OUTPUT_DIR = DATA / 'classify'

# ---------- SETTINGS ----------
SEED = 37
MIN_WORDS = 50
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

LABEL_HUMAN = 0
LABEL_AI = 1
COLUMNS = ('id', 'prompt_id', 'model', 'outlet', 'label', 'n_words', 'text')
# Applied to both classes. Typographic punctuation is a publisher habit, not an
# authorship signal, if it is not normalised, it can be easily detected.
_TYPOGRAPHIC = str.maketrans(
    {'’': "'", '‘': "'", '“': '"', '”': '"', '—': '-', '–': '-', '…': '...'}
)
WHITESPACE = re.compile(r'\s+')


def harmonise(text: str) -> str:
    """Collapse whitespace and normalise punctuation.

    Args:
        text (str): Raw article or generated text.

    Returns:
        str: Cleaned text.

    """
    return WHITESPACE.sub(' ', str(text)).translate(_TYPOGRAPHIC).strip()


def _read_human() -> pd.DataFrame:
    """Return the held-out Irish human anchor.

    Returns:
        pd.DataFrame: One row per usable anchor article.

    Raises:
        FileNotFoundError: If the parsed anchor is absent.

    """
    if not ANCHOR.exists():
        message = f'{ANCHOR} not found; the anchor is local-only.'
        raise FileNotFoundError(message)

    frame = pd.read_csv(
        ANCHOR, usecols=['article_id', 'outlet', 'body_text'], dtype={'article_id': str}
    )
    frame['text'] = frame['body_text'].map(harmonise)
    frame['n_words'] = frame['text'].str.split().str.len()
    frame = frame[frame['n_words'] >= MIN_WORDS].copy()
    frame['id'] = 'anchor:' + frame['outlet'] + ':' + frame['article_id']
    frame['model'] = 'human'
    frame['label'] = LABEL_HUMAN
    frame['prompt_id'] = ''
    logger.info('anchor articles: %d', len(frame))
    return frame[list(COLUMNS)]


def _read_ai(target_total: int, seed: int = SEED) -> pd.DataFrame:
    """Return an AI sample balanced against the human class and even across models.

    ``known_ai.csv`` ids are ``<model>:<prompt_id>``, which is what lets an AI row be
    tied back to the prompt its human counterpart came from.

    Args:
        target_total (int): Number of AI rows wanted in total.
        seed (int): RNG seed, so the sample is reproducible.

    Returns:
        pd.DataFrame: The sampled AI rows.

    Raises:
        FileNotFoundError: If ``known_ai.csv`` is absent.

    """
    if not KNOWN_AI.exists():
        message = f'{KNOWN_AI} not found. Run `python -m calibrate.clean_gsingh` first.'
        raise FileNotFoundError(message)

    frame = pd.read_csv(KNOWN_AI)
    frame['text'] = frame['text'].astype(str).map(harmonise)
    frame['n_words'] = frame['text'].str.split().str.len()
    frame = frame[frame['n_words'] >= MIN_WORDS].copy()
    frame['prompt_id'] = frame['id'].str.rsplit(':', n=1).str[-1].astype(int)

    models = sorted(frame['model'].unique())
    per_model = target_total // len(models)
    rng = np.random.default_rng(seed)
    sampled = [
        rows.iloc[rng.permutation(len(rows))[: min(per_model, len(rows))]]
        for _, rows in frame.groupby('model', sort=True)
    ]
    ai = pd.concat(sampled, ignore_index=True)
    ai['label'] = LABEL_AI
    logger.info(
        'AI rows sampled: %d across %d models (%d each)', len(ai), len(models), per_model
    )
    return ai[['id', 'prompt_id', 'model', 'label', 'n_words', 'text']]


def _assign_splits(prompt_ids: np.ndarray, seed: int = SEED) -> dict[int, str]:
    """Return a split label for every prompt id.

    Args:
        prompt_ids (np.ndarray): Unique prompt ids present in the data.
        seed (int): RNG seed, so the partition is reproducible.

    Returns:
        dict[int, str]: ``{prompt_id: 'train' | 'validation' | 'test'}``.

    """
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(np.sort(prompt_ids))
    train_end = int(len(shuffled) * TRAIN_FRACTION)
    validation_end = train_end + int(len(shuffled) * VALIDATION_FRACTION)
    assignment = {}
    for index, prompt_id in enumerate(shuffled):
        if index < train_end:
            assignment[int(prompt_id)] = 'train'
        elif index < validation_end:
            assignment[int(prompt_id)] = 'validation'
        else:
            assignment[int(prompt_id)] = 'test'
    return assignment


def build(output_dir: Path = OUTPUT_DIR, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Build the three splits and write them as CSV.

    Args:
        output_dir (Path): Directory to write ``train.csv`` and friends into.
        seed (int): RNG seed for both the AI sample and the prompt partition.

    Returns:
        dict[str, pd.DataFrame]: The three splits, keyed by name.

    """
    human = _read_human()
    ai = _read_ai(target_total=len(human), seed=seed)
    combined = pd.concat([human, ai], ignore_index=True)[list(COLUMNS)]

    assignment = _assign_splits(combined['prompt_id'].unique(), seed=seed)
    combined['split'] = combined['prompt_id'].map(assignment)

    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {}
    for name in ('train', 'validation', 'test'):
        rows = combined[combined['split'] == name].drop(columns='split')
        rows = rows.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        rows.to_csv(output_dir / f'{name}.csv', index=False)
        splits[name] = rows
        logger.info(
            '%-11s %6d rows  (%d human / %d AI)  %d prompts',
            name,
            len(rows),
            int((rows['label'] == LABEL_HUMAN).sum()),
            int((rows['label'] == LABEL_AI).sum()),
            rows['prompt_id'].nunique(),
        )
    # Overlap suggested by AI
    overlap = set(splits['train']['prompt_id']) & set(splits['test']['prompt_id'])
    if overlap:
        logger.error('prompt leakage between train and test: %d prompts', len(overlap))
    else:
        logger.info('no prompt overlap between splits')
    return splits


def main() -> int:
    """Build the classifier splits.

    Returns:
        int: 0 on success, 1 if a required input is missing.

    """
    try:
        build()
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
