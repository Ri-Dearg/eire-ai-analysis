"""Sample the inventory for each outlet to get a final set of articles for analysis."""

import logging

from .sampler import main

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )

    main()
