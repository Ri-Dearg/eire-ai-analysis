import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
CALIB_DIR = DATA / 'calibration'
DET_DIR = DATA / 'detection'
INPUTS = DET_DIR / 'score_inputs.csv'
CORPUS = DATA / 'corpus.csv'


def build_inputs() -> Path:
    """Assemble the id -> text table to score.

    Returns:
        Path: The written score_inputs.csv.

    """
    DET_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    corpus = pd.read_csv(CORPUS, dtype=str).fillna('')
    for _, row in corpus.iterrows():
        rows.append(
            {
                'id': f'corpus:{row["article_id"]}',
                'group': 'corpus',
                'outlet': row['outlet'],
                'model': '',
                'period': row['period'],
                'is_wire': row['is_wire'],
                'word_count': row['word_count'],
                'text': row['body_text'],
            }
        )

    out = pd.DataFrame(
        rows,
        columns=[
            'id',
            'group',
            'outlet',
            'model',
            'period',
            'is_wire',
            'word_count',
            'text',
        ],
    )
    out.to_csv(INPUTS, index=False)
    logger.info(
        'wrote %d score inputs %d corpus -> %s',
        len(out),
        (out.group == 'corpus').sum(),
        INPUTS,
    )
    return INPUTS
