"""WIP."""

import logging
import re
import time
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------- CONFIG ----------
USER_AGENT = (
    'CapstoneResearchBot/0.1 (HDip Data Analytics, DBS; contact: 20074605@mydbs.ie)'
)
HEADERS = {'User-Agent': USER_AGENT}
# Request timeout in seconds.
REQUEST_TIMEOUT = 20

# Small delay between sitemap fetches to the same outlet.
INTER_REQUEST_DELAY = 0.5

# Standard sitemap locations to try, in order (same as the probe).
SITEMAP_CANDIDATES = (
    '/sitemap_index.xml',
    '/sitemap.xml',
    '/sitemaps/sitemap.xml',
    '/sitemap-index/44-google_sitemap.xml',
    '/wp-sitemap.xml',
    '/news-sitemap.xml',
)

# HTTP status that counts as a successful fetch.
HTTP_OK = 200


@dataclass(frozen=True)
class Outlet:
    """Configuration for outlet details.

    Attributes:
        slug (str): DB outlet name, passed to ingest(outlet_name=...).
        base_url (str): Outlet root used to locate the sitemap.
        article_re (re.Pattern[str]): A URL must match this to count as a
            sampleable article.

    """

    slug: str
    base_url: str
    article_re: re.Pattern[str]


# RTE news articles take two URL shapes, both of which we want:
# news/business/2026/0331/1566044-...
# no category, e.g. /news/2026/0330/1565927-...
# The category segment is therefore optional. This deliberately excludes
# /sport/, /radio/, /entertainment/, /lifestyle/, /culture/, /brainstorm/ etc.
_RTE_NEWS_RE = re.compile(r'^https?://(?:www\.)?rte\.ie/news/(?:[^/]+/)?\d{4}/\d{4}/\d+')

OUTLETS: dict[str, Outlet] = {
    'rte': Outlet(
        slug='rte',
        base_url='https://www.rte.ie',
        article_re=_RTE_NEWS_RE,
    ),
}


# ---------- FETCH / PARSE ----------
def fetch_xml(url: str) -> str | None:
    """Fetch a URL and return its body if it looks like sitemap XML.

    Args:
        url (str): URL to fetch.

    Returns:
        str | None: Response body if it is sitemap XML, else None.

    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    except requests.RequestException:
        logger.warning('fetch failed: %s', url)
        return None
    if resp.status_code != HTTP_OK:
        logger.warning('Non-OK status code %d found at %s', resp.status_code, url)
        return None
    if '<urlset' in resp.text or '<sitemapindex' in resp.text:
        logger.info('XML found at %s', url)
        return resp.text
    return None


def find_sitemap(base_url: str) -> str | None:
    """Try common sitemap locations until one returns sitemap XML.

    Args:
        base_url (str): Outlet root URL.

    Returns:
        str | None: The sitemap XML, or None if none of the candidate locations returned sitemap XML.

    """
    for path in SITEMAP_CANDIDATES:
        url = base_url.rstrip('/') + path
        xml = fetch_xml(url)
        if xml:
            logger.info('sitemap found: %s', url)
            return xml
        time.sleep(INTER_REQUEST_DELAY)
    return None


def parse_sitemap(xml: str) -> list[str]:
    """Extract the <loc> URLs from a sitemap or a sitemap index.

    Args:
        xml (str): Raw sitemap XML.

    Returns:
        list[str]: The <loc> URLs (sub-sitemap URLs or article URLs).

    """
    soup = BeautifulSoup(xml, 'xml')
    return [loc.text.strip() for loc in soup.find_all('loc')]


# ---------- COLLECT ----------
def collect(outlet: Outlet):

    top_xml = find_sitemap(outlet.base_url)
    if not top_xml:
        logger.error('no sitemap for %s at standard locations', outlet.slug)
        return []
    top_urls = parse_sitemap(top_xml)
    return top_urls
