"""Fine-tune DeBERTa-v3-base to separate human from AI-generated news text."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT / 'data' / 'classify'
MODEL_DIR = ROOT / 'models' / 'deberta-v3-base-ai-detect'


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
