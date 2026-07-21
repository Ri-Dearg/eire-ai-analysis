"""Run the detector ensemble over the assembled score inputs (checkpointed).

Reads data/detection/score_inputs.csv and, for each detector, writes
data/detection/<detector>.csv
"""

import logging
import sys
from pathlib import Path

import pandas as pd

from detect.detect import DETECTORS, build_detector, run_detector, select_device

logger = logging.getLogger(__name__)

# ---------- DIRECTORIES ----------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
DET_DIR = DATA / 'detection'
INPUTS = DET_DIR / 'score_inputs.csv'
CORPUS = DATA / 'corpus.csv'


# ---------- BUILD TRAINING SET ----------
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

    output = pd.DataFrame(
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
    output.to_csv(INPUTS, index=False)
    logger.info(
        'wrote %d score inputs %d corpus -> %s',
        len(output),
        (output.group == 'corpus').sum(),
        INPUTS,
    )
    return INPUTS


def main() -> int:
    """Score the inputs with the named detectors (default: all).

    Returns:
        int: 0 on success.

    """
    if not INPUTS.exists():
        build_inputs()
    names = list(DETECTORS)
    frame = pd.read_csv(INPUTS, dtype=str).fillna('')
    ids = frame['id'].tolist()
    texts = frame['text'].tolist()
    logger.info('device=%s  inputs=%d  detectors=%s', select_device(), len(ids), names)
    for name in names:
        logger.info('[%s] loading weights + scoring ...', name)
        detector = build_detector(name)
        out = run_detector(detector, ids, texts, DET_DIR / f'{name}.csv')
        logger.info('[%s] done -> %s', name, out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
