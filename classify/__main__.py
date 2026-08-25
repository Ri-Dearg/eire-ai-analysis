"""Classifier stage entry point -- build the splits, fine-tune, then evaluate."""

import logging

from .dataset import main as dataset_main
from .evaluate import main as evaluate_main
from .train import main as train_main

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )
    raise SystemExit(dataset_main() or train_main() or evaluate_main())
