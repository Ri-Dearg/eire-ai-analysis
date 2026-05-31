"""WIP."""

import logging
import re
import time
from dataclasses import dataclass

import requests

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


def find_sitemap(base_url: str) -> tuple[str | None, str | None]:
    """Try common sitemap locations until one returns sitemap XML.

    Args:
        base_url (str): Outlet root URL.

    Returns:
        tuple[str | None, str | None]: The sitemap (url, xml), or (None, None)
            if none of the candidate locations returned sitemap XML.

    """
    for path in SITEMAP_CANDIDATES:
        url = base_url.rstrip('/') + path
        xml = fetch_xml(url)
        if xml:
            logger.info('sitemap found: %s', url)
            return url, xml
        time.sleep(INTER_REQUEST_DELAY)
    return None, None
