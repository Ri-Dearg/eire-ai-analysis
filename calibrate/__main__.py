"""Calibration stage entry point -- runs the repeatable calibration only."""

import logging

from .calibrate_select import main as select_main

# LOGGING
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Basic logging setup.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )
    raise SystemExit(select_main())
