"""Module parses and curates collected raw articles."""

import logging

from .curate import main as curator_main
from .parse import main as parser_main

# LOGGING
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Basic logging setup.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )
    parser_main()
    curator_main()
