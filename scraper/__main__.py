"""Provide a sample of urls to scrape and ingest into the database."""

import logging

from .scraper import main

# LOGGING
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Basic logging setup.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )
    main()
