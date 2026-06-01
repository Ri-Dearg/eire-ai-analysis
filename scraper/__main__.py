"""Provide a sample of urls to scrape and ingest into the database."""

import logging

from scraper import ingest

# LOGGING
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Basic logging setup.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )
    sample = [
        'https://www.rte.ie/news/business/2026/0331/1566027-unilever-nears-deal-to-merge-foods-unit-with-mccormick/',
        'https://www.rte.ie/news/business/2026/0331/1566035-business-post-appoints-mark-beard-as-ceo/',
        'https://www.rte.ie/news/business/2026/0331/1566030-heathrow-airports-fees-set-to-rise-by-1/',
        'https://www.rte.ie/news/ireland/2026/0330/1565830-my-lovely-horse-rescue/',
        'https://www.rte.ie/news/world/2026/0330/1565833-cuba-russia-us/',
        'https://www.rte.ie/news/business/2026/0331/1566020-mortgage-approvals-reached-almost-12-billion-in-february/',
        'https://www.rte.ie/brainstorm/2026/0331/1566019-iran-strait-of-harmuz-defence-strategy-persian-gulf/',
        'https://www.rte.ie/news/munster/2026/0330/1565972-cock-fighting/',
        'https://www.rte.ie/news/ulster/2026/0331/1566011-marian-beattie-appeal/',
        'https://www.rte.ie/news/ulster/2026/0330/1565902-schwarzenegger-belfast-honour/',
        'https://www.rte.ie/brainstorm/2026/0324/1565008-early-education-creche-children-education-care-problems/',
        'https://www.rte.ie/news/ireland/2026/0330/1565866-antoin-duffy-court/',
        'https://www.rte.ie/news/dublin/2026/0331/1565981-dublin-property-prices/',
        'https://www.rte.ie/news/middle-east/2026/0330/1565832-iran-war/',
        'https://www.rte.ie/news/newslens/2026/0330/1566004-air-canada-ceo/',
    ]
    ingest(sample, outlet_name='rte', source_feed='sitemap')
