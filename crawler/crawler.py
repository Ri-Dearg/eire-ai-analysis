"""WIP."""

import logging
import re
import time
from dataclasses import dataclass
from datetime import date

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

# Publication date embedded in a URL path: /YYYY/MMDD/.
_URL_DATE_RE = re.compile(r'/(\d{4})/(\d{2})(\d{2})/')

# RTE news articles take two URL shapes, both of which we want:
# news/business/2026/0331/1566044-...
# no category, e.g. /news/2026/0330/1565927-...
# The category segment is therefore optional. This deliberately excludes
# /sport/, /radio/, /entertainment/, /lifestyle/, /culture/, /brainstorm/ etc.
_RTE_NEWS_RE = re.compile(
    r'^https?://(?:www\.)?rte\.ie/news/(?!nuacht/)(?:[^/]+/)?\d{4}/\d{4}/\d+'
)

OUTLETS: dict[str, Outlet] = {
    'rte': Outlet(
        slug='rte',
        base_url='https://www.rte.ie',
        article_re=_RTE_NEWS_RE,
    ),
}


@dataclass(frozen=True)
class Article:
    """A single sampleable article and its sampling metadata.

    Attributes:
        url (str): The article URL as found in the sitemap (passed to ingest()).
        dedup_key (str): Lightly normalised URL used to collapse in-sample
            duplicates. Authoritative canonicalisation happens in raw_scraper.
        pub_date (date): Publication date.
        period (str): 'pre' or 'post', relative to the ChatGPT release.

    """

    url: str
    dedup_key: str
    pub_date: date
    period: str


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
def _fetch_xml(url: str) -> str | None:
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


def _find_sitemap(base_url: str) -> str | None:
    """Try common sitemap locations until one returns sitemap XML.

    Args:
        base_url (str): Outlet root URL.

    Returns:
        str | None: The sitemap XML, or None if none of the candidate locations returned sitemap XML.

    """
    for path in SITEMAP_CANDIDATES:
        url = base_url.rstrip('/') + path
        xml = _fetch_xml(url)
        if xml:
            logger.info('sitemap found: %s', url)
            return xml
        time.sleep(INTER_REQUEST_DELAY)
    return None


def _parse_sitemap(xml: str) -> list[str]:
    """Extract the <loc> URLs from a sitemap or a sitemap index.

    Args:
        xml (str): Raw sitemap XML.

    Returns:
        list[str]: The <loc> URLs (sub-sitemap URLs or article URLs).

    """
    soup = BeautifulSoup(xml, 'xml')
    return [loc.text.strip() for loc in soup.find_all('loc')]


# ---------- SORT ----------
def _clean_url(url: str) -> str:
    """Return a cleaned version of the URL for deduplication.

    Args:
        url (str): URL from the sitemap.

    Returns:
        str: Normalised key.

    """
    return url.strip().split('#', 1)[0].rstrip('/')


def _date_from_url(loc: str) -> date | None:
    """Extract the publication date from a URL path.

    Args:
        loc (str): Article URL.

    Returns:
        date | None: Parsed publication date.

    """
    date_string = _URL_DATE_RE.search(loc)
    if not date_string:
        return None
    try:
        return date(int(date_string[1]), int(date_string[2]), int(date_string[3]))
    except ValueError:
        return None


def _is_article(url: str, outlet: Outlet) -> bool:
    """Decide whether a URL is a sampleable.

    Args:
        url (str): URL to test.
        outlet (Outlet): Outlet configuration providing the filter.

    Returns:
        bool: True if the URL should be sampled.

    """
    return outlet.article_re.match(url) is not None


# ---------- COLLECT ----------
def collect(outlet: Outlet, max_sub_sitemaps: int | None = None):

    top_xml = _find_sitemap(outlet.base_url)
    if not top_xml:
        logger.error('no sitemap for %s at standard locations', outlet.slug)
        return []
    top_urls = _parse_sitemap(top_xml)
    sub_urls = [url for url in top_urls if url.endswith('.xml')]
    direct_links = [url for url in top_urls if not url.endswith('.xml')]

    if max_sub_sitemaps is not None:
        sub_urls = sub_urls[:max_sub_sitemaps]

    total = len(sub_urls)

    for i, sm_url in enumerate(sub_urls, 1):
        time.sleep(INTER_REQUEST_DELAY)
        xml = _fetch_xml(sm_url)
        if not xml:
            logger.warning('%d/%d failed: %s', i, total, sm_url)
            continue
        xml_urls = _parse_sitemap(xml)
        print(xml_urls)
        for url in xml_urls:
            if _is_article(url, outlet):
                clean_url = _clean_url(url)
                print(_date_from_url(url), clean_url)

    return None


collect(OUTLETS['rte'], max_sub_sitemaps=2)
