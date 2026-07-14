"""Gript ingest via the WordPress REST API instead of the article page.

Why this module exists:
    Gript article pages render the body with JavaScript, so a
    requests fetch stores the page chrome and <head> meta but no
    body text. The scraper.ingest captured bodyless Gript rows.
    WordPress exposes the full body server-side.

    This module reuses every `scraper.scraper` helper (DB connection, dedup,
    canonicalisation, storage) and only swaps the fetch step.
"""

import json
import logging
import random
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup

from scraper.scraper import (
    DATA_DIR,
    DB_PATH,
    DELAY_RANGE,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    RETRY_STATUSES,
    _already_have,
    _canonic_url,
    _connect,
    _create_session,
    _outlet_id,
    _pause,
    _read_sample,
    _retry_after,
    _store_page,
)

logger = logging.getLogger(__name__)

# Outlet slug (must match the DB `outlet.name` and the sample file prefix).
GRIPT_SLUG = 'gript'
# WordPress REST collection endpoint for posts.
REST_BASE = 'https://gript.ie/wp-json/wp/v2/posts'
# Provenance tag stored on re-ingested rows (vs the earlier 'sitemap' rows).
SOURCE_FEED = 'wp-rest'
# Persistent log of premium canonical URLs. Premium posts are never
# stored, so _already_have can't suppress them — this log lets re-runs skips
# them before fetching.
PREMIUM_LOG = DATA_DIR / 'gript_premium.log'
# Fields requested: full body + parse metadata, dropping jetpack/related/_links
# bloat that the default response carries.
_REST_FIELDS = (
    'id',
    'date',
    'date_gmt',
    'modified',
    'slug',
    'link',
    'status',
    'title',
    'content',
    'excerpt',
    'categories',
    'tags',
    'yoast_head_json',
)
HTTP_OK = 200
VERIFY_DEFAULT = 8
# Below this many chars of stripped body, treat a REST result as suspect.
MIN_BODY_CHARS = 200
# Markers identifying a Memberful premium teaser: the REST API returns only an
# intro + a 'subscribe' block, not the full body.
PREMIUM_MARKERS = (
    'memberful-global-teaser-content',
    'This article is premium content',
)


# ---------- URL → REST ----------
def _slug_from_url(url: str) -> str:
    """Derive the WordPress post slug from a Gript article URL.

    Args:
        url (str): Article URL.

    Returns:
        str: The trailing path segment.

    """
    return _canonic_url(url).rstrip('/').rsplit('/', 1)[-1]


def _rest_url(slug: str) -> str:
    """Build the REST query URL for a single post slug.

    Args:
        slug (str): Post slug.

    Returns:
        str: Fully-formed query URL.

    """
    query = urlencode({'slug': slug, '_fields': ','.join(_REST_FIELDS)})
    return f'{REST_BASE}?{query}'


# ---------- FETCH ----------
def _post_payload(
    resp: requests.Response,
    article_url: str,
) -> tuple[int, str, str] | None:
    """Turn a REST response into a storable (status, json, url) triple.

    The stored body is the single post object as  JSON.

    Args:
        resp (requests.Response): The REST API response.
        article_url (str): The original article URL.

    Returns:
        tuple[int, str, str] | None: (status, payload, article_url) on a
            usable post, else None.

    """
    if resp.status_code != HTTP_OK:
        logger.warning('rest %s for %s', resp.status_code, article_url)
        return None
    try:
        posts = resp.json()
    except ValueError:
        logger.warning('non-JSON rest response for %s', article_url)
        return None
    if not posts:
        logger.warning('no post for slug: %s', article_url)
        return None
    payload = json.dumps(posts[0], ensure_ascii=False)
    return resp.status_code, payload, article_url


def _fetch_post(
    session: requests.Session,
    article_url: str,
) -> tuple[int, str, str] | None:
    """Fetch a single Gript post via REST, with the same retry policy as `_fetch`.

    Args:
        session (requests.Session): User-agent session.
        article_url (str): Article URL.

    Returns:
        tuple[int, str, str] | None: Storable triple, or None on failure.

    """
    rest_url = _rest_url(_slug_from_url(article_url))
    for attempt in range(MAX_RETRIES + 1):
        session.cookies.clear()
        try:
            resp = session.get(rest_url, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                logger.exception('transport error %s', rest_url)
                return None
            wait = _pause(attempt)
            logger.warning(
                'retry %d/%d in %.1fs (%s)', attempt + 1, MAX_RETRIES, wait, exc
            )
            time.sleep(wait)
            continue

        if resp.status_code in RETRY_STATUSES and attempt < MAX_RETRIES:
            wait = _retry_after(resp) or _pause(attempt)
            logger.warning(
                '%s %s; retry %d/%d in %.1fs',
                resp.status_code,
                rest_url,
                attempt + 1,
                MAX_RETRIES,
                wait,
            )
            time.sleep(wait)
            continue

        return _post_payload(resp, article_url)
    return None
