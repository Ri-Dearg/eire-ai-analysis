"""Score the fine-tuned classifier: held-out F1, per-generator recall, corpus FPR."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'


def main(argv: list[str] | None = None) -> int:
    """Evaluate the classifier and write ``classifier_report.csv``.

    Args:
        argv (list[str] | None): Command-line arguments; ``None`` uses ``sys.argv``.

    Returns:
        int: 0 on success, 1 if a required input is missing.

    """
    try:
        logger.exception('missing input')
        return 1
    logger.info('wrote %s', REPORT.name)
    return 0

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )
    sys.exit(main())
