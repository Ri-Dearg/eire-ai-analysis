"""Run the detector ensemble over the assembled score inputs (checkpointed).

Reads data/detection/score_inputs.csv and, for each detector, writes
data/detection/<detector>.csv
"""

import logging
import sys
from pathlib import Path

import pandas as pd
from calibrate.calibrate import INPUTS, build_inputs

from detect.detect import DETECTORS, build_detector, run_detector, select_device

logger = logging.getLogger(__name__)

# ---------- DIRECTORIES ----------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
DET_DIR = DATA / 'detection'
CORPUS = DATA / 'corpus.csv'


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
        output = run_detector(detector, ids, texts, DET_DIR / f'{name}.csv')
        logger.info('[%s] done -> %s', name, output)
    return 0


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )
    sys.exit(main())
