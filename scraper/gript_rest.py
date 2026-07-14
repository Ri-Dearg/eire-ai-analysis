"""Gript ingest via the WordPress REST API instead of the article page.

Why this module exists:
    Gript article pages render the body with JavaScript, so a
    requests fetch stores the page chrome and <head> meta but no
    body text. The scraper.ingest captured
    bodyless Gript rows. WordPress exposes the full body server-side.

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
