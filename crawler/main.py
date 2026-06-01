"""Functions to crawl sitemaps, extract article URLs with publication dates."""

from __future__ import annotations

import csv
import logging
import re
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from bs4 import BeautifulSoup

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

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

CHATGPT_RELEASE = date(2022, 11, 30)

# RETRIES
# Retry transient failures: transport errors + these HTTP statuses.
MAX_RETRIES = 3
RETRY_STATUSES = {429, 500, 502, 503, 504}
PAUSE_BASE = 2.0
PAUSE_CAP = 30.0

# SITE SETTINGS
SUB_SITEMAP_INCLUDE = {
    'rte': None,  # None = follow all sub-sitemaps (current RTE behaviour)
    'gript': re.compile(r'/post-sitemap\d*\.xml$'),  # only post-sitemaps
}

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

# Non-article wordpress URL segments to exclude.
_NON_ARTICLE_RE = re.compile(
    r'/(category|tag|author|page|wp-content|wp-includes|feed|comments|'
    r'about|contact|privacy|terms|advertise|subscribe|topic|section)/'
)


@dataclass(frozen=True)
class Article:
    """A single sampleable article and its data.

    Attributes:
        url (str): The article URL as found in the sitemap.
        clean_url (str): Lightly normalised URL used to prevent duplicates.
            Authoritative canonicalisation happens in raw_scraper.
        pub_date (date): Publication date.
        period (str): 'pre' or 'post', relative to the ChatGPT release.

    """

    url: str
    clean_url: str
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
    date_source: str
    article_re: re.Pattern[str] | None = None


OUTLETS: dict[str, Outlet] = {
    'rte': Outlet(
        slug='rte',
        base_url='https://www.rte.ie',
        date_source='url',
        article_re=_RTE_NEWS_RE,
    ),
    'gript': Outlet(
        slug='gript',
        base_url='https://gript.ie',
        date_source='lastmod',
        article_re=None,
    ),
}


# ---------- FETCH / PARSE ----------
def _fetch_xml(url: str) -> str | None:
    """Fetch a URL and return its body if it looks like sitemap XML.

    Args:
        url (str): URL to fetch.

    Returns:
        str | None: Response body if it is sitemap XML, else None.

    """
    # Run through URls and get the XML. Log issues.
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        except requests.RequestException:
            if attempt == MAX_RETRIES:
                logger.warning('fetch failed after %d retries: %s', MAX_RETRIES, url)
                return None
            wait = min(PAUSE_BASE * 2**attempt, PAUSE_CAP)
            logger.info('transport error, retrying in %.1fs: %s', wait, url)
            time.sleep(wait)
            continue
        if resp.status_code != HTTP_OK:
            if resp.status_code in RETRY_STATUSES and attempt < MAX_RETRIES:
                wait = min(PAUSE_BASE * 2**attempt, PAUSE_CAP)
                logger.info('%d at %s, retrying in %.1fs', resp.status_code, url, wait)
                time.sleep(wait)
                continue
            logger.warning('Non-OK status code %d found at %s', resp.status_code, url)
            return None
        if '<urlset' in resp.text or '<sitemapindex' in resp.text:
            logger.info('XML found at %s', url)
            return resp.text
        return None
    return None


def _find_sitemap(base_url: str) -> str | None:
    """Try common sitemap locations until one returns sitemap XML.

    Args:
        base_url (str): Outlet root URL.

    Returns:
        str | None: The sitemap XML, or None if no sitemap XML.

    """
    # Run through candidate sitemap paths.
    for path in SITEMAP_CANDIDATES:
        url = base_url.rstrip('/') + path
        xml = _fetch_xml(url)
        if xml:
            logger.info('sitemap found: %s', url)
            return xml
        time.sleep(INTER_REQUEST_DELAY)
    return None


def _parse_sitemap(xml: str) -> list[dict[str, str]]:
    """Extract the <loc> URLs from a sitemap or a sitemap index.

    Args:
        xml (str): Raw sitemap XML.

    Returns:
        list[dict[str, str]]: The <loc> and <lastmod> values.

    """
    soup = BeautifulSoup(xml, 'xml')
    xml_keys: list[dict[str, str]] = []
    for element in soup.find_all(['url', 'sitemap']):
        loc = element.find('loc')
        if not loc:
            continue
        lastmod = element.find('lastmod')
        xml_keys.append(
            {
                'loc': loc.text.strip(),
                'lastmod': lastmod.text.strip() if lastmod else '',
            }
        )
    return xml_keys


# ---------- ORGANISE ----------
def _clean_url(url: str) -> str:
    """Return a cleaned version of the URL for deduplication.

    Args:
        url (str): URL from the sitemap.

    Returns:
        str: Normalised key.

    """
    return url.strip().split('#', 1)[0].rstrip('/')


def date_from_lastmod(lastmod: str) -> date | None:
    """Parse a sitemap <lastmod> value into a date.

    Args:
        lastmod (str): The <lastmod> string (may be empty).

    Returns:
        date | None: Parsed date, or None if absent or invalid.

    """
    if len(lastmod) < len('YYYY-MM-DD'):
        return None
    try:
        return date(int(lastmod[:4]), int(lastmod[5:7]), int(lastmod[8:10]))
    except ValueError:
        return None


def _date_from_url(loc: str) -> date | None:
    """Extract the publication date from a URL path.

    Args:
        loc (str): Article URL.

    Returns:
        date | None: Parsed publication date.

    """
    # Use regex to find the date segments in the URL.
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
    if outlet.article_re is not None:
        return outlet.article_re.match(url) is not None
    return _NON_ARTICLE_RE.search(url) is None


def _urls_to_articles(
    entries: list[dict[str, str]],
    outlet: Outlet,
    captured: set[str],
) -> list[Article]:
    """Filter sitemap URLs down to dated, cleaned articles.

    Args:
        entries (list[dict[str, str]]): Parsed sitemap entries.
        outlet (Outlet): Outlet configuration for filtering and dating.
        captured (set[str]): Clean keys already taken. Mutated in place.

    Returns:
        list[Article]: New articles found in these URLs.

    """
    output_articles: list[Article] = []
    for entry in entries:
        loc = entry['loc']
        # Check if the URL is an article.
        if not _is_article(loc, outlet):
            continue
        # Clean the URL and check for duplicates.
        clean_url = _clean_url(loc)
        if clean_url in captured:
            continue
        # Extract the publication date from the URL.
        if outlet.date_source == 'url':
            pub = _date_from_url(loc)
        else:
            pub = date_from_lastmod(entry.get('lastmod', ''))
        if pub is None:
            continue
        captured.add(clean_url)
        period = 'pre' if pub < CHATGPT_RELEASE else 'post'
        # Add the article to the output list.
        output_articles.append(
            Article(url=loc, clean_url=clean_url, pub_date=pub, period=period)
        )
    return output_articles


# ---------- COLLECT ----------
def collect(outlet: Outlet, max_sub_sitemaps: int | None = None) -> list[Article]:
    """Walk every sitemap and return all cleaned, dated articles.

    Distinguishes between top-level sitemaps with direct article links and sub-sitemaps.

    Args:
        outlet (Outlet): Outlet to collect.
        max_sub_sitemaps (int | None, optional): Cap on sub-sitemaps
            followed. Defaults to None (unlimited).

    Returns:
        list[Article]: All articles collected for the outlet.

    """
    # Find the top-level sitemap XML.
    top_xml = _find_sitemap(outlet.base_url)
    if not top_xml:
        logger.error('no sitemap for %s at standard locations', outlet.slug)
        return []
    # Parse the top-level sitemap, separate direct article links from sub-sitemap links.
    top_urls = _parse_sitemap(top_xml)
    sub_urls = [url for url in top_urls if url.endswith('.xml')]
    direct_links = [url for url in top_urls if not url.endswith('.xml')]

    # Set up the article list and the captured URL set.
    captured_urls: set[str] = set()
    articles: list[Article] = []

    # Process any direct article links in the top-level sitemap.
    if direct_links:
        logger.info('Direct article links found in top-level sitemap.')
        articles.extend(_urls_to_articles(direct_links, outlet, captured_urls))

    # Process sub-sitemaps, with an optional cap.
    if max_sub_sitemaps is not None:
        sub_urls = sub_urls[:max_sub_sitemaps]

    total = len(sub_urls)
    if total:
        logger.info('Sub-sitemaps found.')
    # Loop through sub-sitemaps, fetch and parse them, and extract urls.
    for i, sitemap_url in enumerate(sub_urls, 1):
        time.sleep(INTER_REQUEST_DELAY)
        xml = _fetch_xml(sitemap_url)
        if not xml:
            logger.warning('%d/%d failed: %s', i, total, sitemap_url)
            continue
        found = _urls_to_articles(_parse_sitemap(xml), outlet, captured_urls)
        articles.extend(found)
        logger.info('%d/%d %s: +%d articles', i, total, sitemap_url, len(found))

    logger.info('collected %d articles for %s', len(articles), outlet.slug)
    return articles


# ---------- OUTPUT ----------
def write_outputs(
    article_url_list: Sequence[Article],
    outlet_slug: str,
    out_dir: str | Path,
) -> tuple[Path, Path]:
    """Write the URL list (.txt) and the .csv.

    Args:
        article_url_list (Sequence[Article]): Articles to write.
        outlet_slug (str): Outlet slug used in the output filenames.
        out_dir (str | Path): Directory to write into; created if needed.

    Returns:
        tuple[Path, Path]: Paths to the written (txt, csv) files.

    """
    # Make the output directory if it doesn't exist.
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Sort the urls by publication date.
    ordered = sorted(
        article_url_list,
        key=lambda article: (article.pub_date, article.url),
        reverse=True,
    )

    # Write the .txt file with one URL per line.
    txt_path = out_dir / f'{outlet_slug}_inventory.txt'
    txt_path.write_text(
        '\n'.join(article.url for article in ordered) + '\n', encoding='utf-8'
    )

    # Write the .csv file with columns.
    csv_path = out_dir / f'{outlet_slug}_inventory.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(('url', 'published_date', 'year', 'period'))
        for article in ordered:
            writer.writerow(
                (
                    article.url,
                    article.pub_date.isoformat(),
                    article.pub_date.year,
                    article.period,
                )
            )

    return txt_path, csv_path


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )

    article_urls = collect(OUTLETS['rte'])
    txt_file, csv_file = write_outputs(article_urls, 'rte', './data/')
    logger.info('wrote %s (%d urls) and %s', txt_file, len(article_urls), csv_file)
