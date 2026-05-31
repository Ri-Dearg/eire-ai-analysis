"""WIP."""

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
    '/sitemap.xml',
    '/sitemaps/sitemap.xml',
    '/sitemap-index/44-google_sitemap.xml',
    '/wp-sitemap.xml',
    '/news-sitemap.xml',
)

# HTTP status that counts as a successful fetch.
HTTP_OK = 200

CHATGPT_RELEASE = date(2022, 11, 30)

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
    article_re: re.Pattern[str]


OUTLETS: dict[str, Outlet] = {
    'rte': Outlet(
        slug='rte',
        base_url='https://www.rte.ie',
        article_re=_RTE_NEWS_RE,
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
    return outlet.article_re.match(url) is not None


def urls_to_articles(
    locs: Iterable[str],
    outlet: Outlet,
    captured: set[str],
) -> list[Article]:
    """Filter sitemap URLs down to dated, cleaned articles.

    Args:
        locs (Iterable[str]): URLs from a sitemap.
        outlet (Outlet): Outlet configuration for filtering and dating.
        captured (set[str]): Clean keys already taken.

    Returns:
        list[Article]: New articles found in these URLs.

    """
    output_articles: list[Article] = []
    for loc in locs:
        # Check if the URL is an article.
        if not _is_article(loc, outlet):
            continue
        # Clean the URL and check for duplicates.
        clean_url = _clean_url(loc)
        if clean_url in captured:
            continue
        # Extract the publication date from the URL.
        pub = _date_from_url(loc)
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
    top_xml = _find_sitemap(outlet.base_url)
    if not top_xml:
        logger.error('no sitemap for %s at standard locations', outlet.slug)
        return []
    top_urls = _parse_sitemap(top_xml)
    sub_urls = [url for url in top_urls if url.endswith('.xml')]
    direct_links = [url for url in top_urls if not url.endswith('.xml')]

    captured_urls: set[str] = set()
    articles: list[Article] = []

    if direct_links:
        logger.info('Direct article links found in top-level sitemap.')
        articles.extend(urls_to_articles(direct_links, outlet, captured_urls))

    if max_sub_sitemaps is not None:
        sub_urls = sub_urls[:max_sub_sitemaps]

    total = len(sub_urls)

    for i, sitemap_url in enumerate(sub_urls, 1):
        logger.info('Sub-sitemaps found.')
        time.sleep(INTER_REQUEST_DELAY)
        xml = _fetch_xml(sitemap_url)
        if not xml:
            logger.warning('%d/%d failed: %s', i, total, sitemap_url)
            continue
        found = urls_to_articles(_parse_sitemap(xml), outlet, captured_urls)
        articles.extend(found)
        logger.info('%d/%d %s: +%d articles', i, total, sitemap_url, len(found))

    logger.info('collected %d articles for %s', len(articles), outlet.slug)
    return articles


# ---------- OUTPUT ----------
def write_outputs(
    sample: Sequence[Article],
    outlet_slug: str,
    out_dir: str | Path,
) -> tuple[Path, Path]:
    """Write the URL list (.txt) and the .csv.

    Args:
        sample (Sequence[Article]): Articles to write.
        outlet_slug (str): Outlet slug used in the output filenames.
        out_dir (str | Path): Directory to write into; created if needed.

    Returns:
        tuple[Path, Path]: Paths to the written (txt, csv) files.

    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        sample, key=lambda article: (article.pub_date, article.url), reverse=True
    )

    txt_path = out_dir / f'{outlet_slug}_inventory.txt'
    txt_path.write_text(
        '\n'.join(article.url for article in ordered) + '\n', encoding='utf-8'
    )

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
    article_urls = collect(OUTLETS['rte'])
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )

    txt_file, csv_file = write_outputs(article_urls, 'rte', './data/')
    logger.info('wrote %s (%d urls) and %s', txt_file, len(article_urls), csv_file)
