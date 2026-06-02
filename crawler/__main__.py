"""Main file to execute crawler. Edit parameters as needed."""

import logging

from crawler.crawler import main

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )

    main()
