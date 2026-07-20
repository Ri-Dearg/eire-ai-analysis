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

from scrape.scrape import (
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


# ---------- PREMIUM SKIP-LOG ----------
def _append_premium(url: str, path: Path = PREMIUM_LOG) -> None:
    """Append a premium canonical URL to the persistent skip-log.

    Args:
        url (str): Canonical URL of the premium post.
        path (Path, optional): Skip-log path. Defaults to PREMIUM_LOG.

    """
    with path.open('a', encoding='utf-8') as handle:
        handle.write(f'{url}\n')


def _is_premium(post: dict) -> bool:
    """Detect a Memberful premium teaser.

    Args:
        post (dict): A decoded WP REST post object.

    Returns:
        bool: True if the post is premium content.

    """
    if any(marker in _rendered(post.get('content')) for marker in PREMIUM_MARKERS):
        return True
    graph = (post.get('yoast_head_json') or {}).get('schema', {}).get('@graph', [])
    return any(
        isinstance(node, dict) and 'Premium' in (node.get('articleSection') or [])
        for node in graph
    )


def _load_premium_log(path: Path = PREMIUM_LOG) -> set[str]:
    """Load canonical URLs of premium posts seen on previous runs.

    Args:
        path (Path, optional): Skip-log path. Defaults to PREMIUM_LOG.

    Returns:
        set[str]: Canonical URLs to skip without re-fetching (empty if absent).

    """
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    }


# ---------- VERIFY ----------
def _rendered(value: object) -> str:
    """Read a WP REST field that may be a {'rendered': ...} dict or a bare string.

    Gript's REST returns content as a {'rendered': ...} object but
    title/excerpt as plain strings, so accessors must tolerate both.

    Args:
        value (object): A post field value (dict, str, or missing).

    Returns:
        str: The rendered text, or '' if absent.

    """
    if isinstance(value, dict):
        return str(value.get('rendered', ''))
    if isinstance(value, str):
        return value
    return ''


def _strip_html(html: str) -> str:
    """Return visible text from HTML.

    Args:
        html (str): HTML.

    Returns:
        str: Plain text.

    """
    return BeautifulSoup(html, 'html.parser').get_text(' ', strip=True)


# Written by AI
def verify(n: int = VERIFY_DEFAULT) -> None:
    """Probe n sampled slugs and report body extraction; no DB writes.

    Args:
        n (int, optional): Number of slugs to probe. Defaults to VERIFY_DEFAULT.

    """
    urls = _read_sample(GRIPT_SLUG, DATA_DIR)[:n]
    if not urls:
        print('no gript sample urls found')
        return
    print(f'Probing {len(urls)} Gript slugs via wp-rest\n')
    ok = premium = 0
    with _create_session() as session:
        for i, url in enumerate(urls, 1):
            result = _fetch_post(session, url)
            if result is None:
                print(f'[{i}] FAILED  {url}')
                continue
            post = json.loads(result[1])
            body = _strip_html(_rendered(post.get('content')))
            title = _rendered(post.get('title'))
            author = (post.get('yoast_head_json') or {}).get('author', '')
            is_prem = _is_premium(post)
            full = not is_prem and len(body) >= MIN_BODY_CHARS
            ok += full
            premium += is_prem
            tag = 'PREMIUM' if is_prem else ('OK   ' if full else 'SHORT')
            print(f'[{i}] {tag} body={len(body):>5}c author={author!r:18} | {title[:48]}')
            if i < len(urls):
                time.sleep(random.uniform(*DELAY_RANGE))
    print(
        f'\n{ok}/{len(urls)} full bodies, {premium} premium teasers '
        f'(gated, not recoverable via REST).'
    )


# ---------- INGEST ----------
def ingest_gript(
    urls: list[str],
    db_path: str = DB_PATH,
    delay_range: tuple[float, float] = DELAY_RANGE,
) -> dict[str, int]:
    """Re-ingest Gript articles via the REST API.

    Mirrors scraper.ingest's outcome but fetches the REST
    endpoint instead of the article page.

    Args:
        urls (list[str]): Sampled Gript article URLs.
        db_path (str, optional): DB path. Defaults to DB_PATH.
        delay_range (tuple[float, float], optional): Delay window.

    Returns:
        dict[str, int]: Counts of stored / skipped / failed / not_stored.

    """
    conn = _connect(db_path)
    try:
        oid = _outlet_id(conn, GRIPT_SLUG)
        counts: Counter[str] = Counter(
            {'stored': 0, 'skipped': 0, 'premium': 0, 'failed': 0, 'not_stored': 0}
        )
        premium_seen = _load_premium_log()
        with _create_session() as session:
            for url in urls:
                canon = _canonic_url(url)
                if _already_have(conn, canon):
                    counts['skipped'] += 1
                    continue
                # Known-premium from a previous run: skip with no fetch/delay.
                if canon in premium_seen:
                    counts['premium'] += 1
                    logger.info('premium (cached): %s', url)
                    continue
                result = _fetch_post(session, url)
                if result is None:
                    counts['failed'] += 1
                    logger.info('failed: %s', url)
                    continue
                # Premium = Memberful teaser only; full body is gated, so skip it
                # Log it so runs skip it before fetching.
                if _is_premium(json.loads(result[1])):
                    _append_premium(canon)
                    premium_seen.add(canon)
                    counts['premium'] += 1
                    logger.info('premium (skipped): %s', url)
                    time.sleep(random.uniform(*delay_range))
                    continue
                outcome = (
                    'stored'
                    if _store_page(conn, oid, SOURCE_FEED, result)
                    else ('not_stored')
                )
                counts[outcome] += 1
                logger.info('%s: %s', outcome, url)
                time.sleep(random.uniform(*delay_range))
    finally:
        conn.close()

    logger.info('gript rest done: %s', dict(counts))
    return dict(counts)


# ---------- PURGE ----------
def purge_gript(db_path: str = DB_PATH) -> int:
    """Delete all existing Gript rows so REST re-ingest isn't skipped by dedup.

    Removes raw_page rows first, then their `article` rows.

    Args:
        db_path (str, optional): DB path. Defaults to DB_PATH.

    Returns:
        int: Number of article rows deleted.

    """
    conn = _connect(db_path)
    try:
        oid = _outlet_id(conn, GRIPT_SLUG)
        with conn:
            conn.execute(
                'DELETE FROM raw_page WHERE article_id IN '
                '(SELECT id FROM article WHERE outlet_id = ?)',
                (oid,),
            )
            cur = conn.execute('DELETE FROM article WHERE outlet_id = ?', (oid,))
        deleted = cur.rowcount
    finally:
        conn.close()
    logger.info('purged %d gript articles', deleted)
    return deleted


def main() -> None:
    """Dispatch on argv: --verify [N] / --purge / bare re-ingest."""
    args = sys.argv[1:]
    if args and args[0] == '--verify':
        verify(int(args[1]) if len(args) > 1 else VERIFY_DEFAULT)
        return
    if args and args[0] == '--purge':
        print(f'purged {purge_gript()} gript rows')
        return
    urls = _read_sample(GRIPT_SLUG, DATA_DIR)
    if not urls:
        print('no gript sample urls found')
        return
    print(f'ingesting {len(urls)} gript urls via wp-rest')
    print(ingest_gript(urls))


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s'
    )
    main()
