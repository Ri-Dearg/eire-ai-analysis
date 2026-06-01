"""Main file to execute crawler. Edit parameters as needed."""

import logging

from crawler.crawler import OUTLETS, collect, write_outputs

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )

    article_urls = collect(OUTLETS['irish_examiner'])
    txt_file, csv_file = write_outputs(article_urls, 'irish_examiner', './data/')
    logger.info('wrote %s (%d urls) and %s', txt_file, len(article_urls), csv_file)
