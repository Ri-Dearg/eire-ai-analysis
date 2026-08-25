"""Build balanced, leakage-free train/validation/test splits for the classifier."""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------- FILE SYSTEM ----------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
ANCHOR = DATA / 'calibration' / 'human_parsed.csv'
GENERATED = (
    DATA / 'generation' / 'generated_irish_ai.csv',
    DATA / 'generation' / 'generated_frontier_ai.csv',
)
OUTPUT_DIR = DATA / 'classify'

# ---------- SETTINGS ----------
SEED = 37
MIN_WORDS = 50
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

LABEL_HUMAN = 0
LABEL_AI = 1
COLUMNS = ('id', 'prompt_id', 'model', 'outlet', 'label', 'n_words', 'text')
# Generated text is trimmed to its last complete sentence, so human must be too.
SENTENCE_END = re.compile(r'^.*[.!?]["\']?', re.DOTALL)
# Applied to both classes. Typographic punctuation is a publisher habit, not an
# authorship signal, if it is not normalised, it can be easily detected.
# ruff: ignore [RUF001]
_TYPOGRAPHIC = str.maketrans(
    {'’': "'", '‘': "'", '“': '"', '”': '"', '—': '-', '–': '-', '…': '...'}
)
WHITESPACE = re.compile(r'\s+')
WORD_BUDGET = 400


def harmonise(text: str, budget: int = WORD_BUDGET) -> str:
    """Collapse whitespace, normalise punctuation, cut to budget, end on a sentence.

    Args:
        text (str): Raw article or generated text.
        budget (int): Maximum words to keep.

    Returns:
        str: Cleaned text, at most ``budget`` words, ending at a sentence boundary.

    """
    cleaned = WHITESPACE.sub(' ', str(text)).translate(_TYPOGRAPHIC).strip()
    cleaned = ' '.join(cleaned.split()[:budget])
    match = SENTENCE_END.match(cleaned)
    return match.group(0) if match else cleaned


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
    """Return a generator-balanced sample of the Irish-register generated set.

    Args:
        target_total (int): Number of AI rows wanted.
        seed (int): RNG seed.

    Returns:
        pd.DataFrame: The sampled generated rows.

    Raises:
        FileNotFoundError: If the generation manifest is absent.

    """
    frames = []
    for path in GENERATED:
        if not path.exists():
            logger.warning('%s not found; skipping', path.name)
            continue
        part = pd.read_csv(path)
        if 'outlet' not in part.columns:
            part['outlet'] = ''
        frames.append(part[['id', 'model', 'outlet', 'text']])

    if not frames:
        message = f'none of {[p.name for p in GENERATED]} found; generate first.'
        raise FileNotFoundError(message)

    frame = pd.concat(frames, ignore_index=True)
    frame['text'] = frame['text'].map(harmonise)
    frame['n_words'] = frame['text'].str.split().str.len()
    frame = frame[frame['n_words'] >= MIN_WORDS].copy()
    frame['prompt_id'] = frame['id'].str.rsplit(':', n=1).str[-1]
    frame['label'] = LABEL_AI

    per_model = max(1, target_total // frame['model'].nunique())
    rng = np.random.default_rng(seed)
    sampled = pd.concat(
        [
            rows.iloc[rng.permutation(len(rows))[: min(per_model, len(rows))]]
            for _, rows in frame.groupby('model', sort=True)
        ],
        ignore_index=True,
    )
    logger.info(
        'generated articles: %d across %d models',
        len(sampled),
        per_model and sampled['model'].nunique(),
    )
    return sampled[list(COLUMNS)]


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
    names = ['train'] * train_end
    names += ['validation'] * (validation_end - train_end)
    names += ['test'] * (len(shuffled) - validation_end)
    return dict(zip(shuffled, names, strict=True))


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

    rng = np.random.default_rng(seed)
    size = min(len(human), len(ai))
    human = human.iloc[rng.permutation(len(human))[:size]]
    combined = pd.concat([human, ai], ignore_index=True)

    assignment = _assign_splits(
        combined.loc[combined['prompt_id'] != '', 'prompt_id'].unique(), seed=seed
    )
    combined['split'] = combined['prompt_id'].map(assignment)

    unassigned = combined['split'].isna()
    combined.loc[unassigned, 'split'] = rng.choice(
        ['train', 'validation', 'test'],
        size=int(unassigned.sum()),
        p=[TRAIN_FRACTION, VALIDATION_FRACTION, 1 - TRAIN_FRACTION - VALIDATION_FRACTION],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {}
    for name in ('train', 'validation', 'test'):
        rows = combined[combined['split'] == name].drop(columns='split')
        rows = rows.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        rows.to_csv(output_dir / f'{name}.csv', index=False)
        splits[name] = rows
        logger.info(
            '%-11s %6d rows  (%d human / %d AI)',
            name,
            len(rows),
            int((rows['label'] == LABEL_HUMAN).sum()),
            int((rows['label'] == LABEL_AI).sum()),
        )
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
