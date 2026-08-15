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

# Fractions of the prompt pool, not of the rows. 70/15/15 leaves ~1,100 prompts in
# each held-out split, which is enough for a stable F1 at this sample size.
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

# NYT prose uses typographic punctuation; LLM output uses ASCII. Left alone the classifier
# learns to distinguish the two.
_TYPOGRAPHIC = str.maketrans(
    {'’': "'", '‘': "'", '“': '"', '”': '"', '—': '-', '–': '-', '…': '...'}
)

# gsingh1's Human_story column carries whole NYT pages, not just articles: video
# transcripts, navigation furniture, section chrome.
_CHROME = re.compile(
    r'(?i)(new video loaded|\btranscript\b|supported by|site search navigation'
    r'|site navigation|advertisement)'
)


def harmonise(text: str) -> str:
    """Apply identical surface cleaning to both classes.

    Args:
        text (str): Raw article text from either class.

    Returns:
        str: Whitespace-collapsed, punctuation-normalised text.

    """
    return normalise(text).translate(_TYPOGRAPHIC)


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
            text = harmonise(row.get(HUMAN_COLUMN) or '')
            if _CHROME.search(text):
                continue
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
