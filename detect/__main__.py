"""AI-generated-text detection stage for the eire-ai-analysis corpus."""

import logging

from .export import main as export_main
from .score import main as score_main

# LOGGING
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Basic logging setup.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )
    raise SystemExit(score_main() or export_main())
