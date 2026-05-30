"""WIP."""

import hashlib
import sqlite3
from datetime import UTC, datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

# Path to DB file.
DB_PATH = './data/dataset.db'

# User agent string for scraper.
USER_AGENT = (
    'CapstoneResearchBot/0.1 (HDip Data Analytics, DBS; contact: 20074605@mydbs.ie)'
)

# Request timeout in seconds.
REQUEST_TIMEOUT = 20

# Polite delay range between requests in seconds (min, max).
DELAY_RANGE = (4.0, 6.0)

# Query parameters to drop during URL canonicalisation.
_TRACKING_PREFIXES = ('utm_',)
_TRACKING_KEYS = {'fbclid', 'gclid', 'mc_cid', 'mc_eid'}


def _canonic_url(url: str) -> str:
    # Canonicalise a URL by normalising scheme.
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()

    # Drop tracking query parameters.
    kept = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key not in _TRACKING_KEYS
        and not any(key.startswith(prefix) for prefix in _TRACKING_PREFIXES)
    ]
    query = urlencode(kept)

    # Normalise path by stripping trailing slash (except for root).
    path = parts.path
    if path.endswith('/') and path != '/':
        path = path.rstrip('/')

    return urlunsplit((scheme, netloc, path, query, ''))


def _connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Open a connection with foreign keys enforced.

    Args:
        db_path (str, optional): DB location. Defaults to DB_PATH.

    Returns:
        sqlite3.Connection: Connection to DB.

    """
    # Connect to DB and enforce foreign keys.
    try:
        conn = sqlite3.connect(db_path)
        conn.execute('PRAGMA foreign_keys = ON')
        conn.execute('PRAGMA integrity_check')
        print(f'Connected to {db_path}')
    except sqlite3.OperationalError as e:
        print(f'Failed to connect to {db_path}: {e}')
        raise
    return conn


def _outlet_id(conn: sqlite3.Connection, name: str) -> int:
    """Get outlet ID from DB.

    Args:
        conn (sqlite3.Connection): DB connection.
        name (str): Name of outlet in DB.

    Raises:
        ValueError: If name is not in DB.

    Returns:
        int: ID of outlet ind DB.

    """
    # Query DB for outlet ID by name.
    row = conn.execute('SELECT id FROM outlet WHERE name = ?', (name,)).fetchone()
    if row is None:
        print(f'outlet {name!r} not found in DB.')
        raise ValueError
    return row[0]


def _create_session() -> requests.Session:
    # Create a requests session with a custom user agent.
    session = requests.Session()
    session.headers.update({'User-Agent': USER_AGENT})
    return session


def _fetch(session: requests.Session, url: str) -> tuple[int, str, str] | None:
    # Fetch a URL with error handling and timeout.
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        print(f'  ! transport error {url}: {e}')
        return None
    return resp.status_code, resp.text, resp.url


def _already_have(conn: sqlite3.Connection, url_canonical: str) -> bool:
    return (
        conn.execute(
            'SELECT 1 FROM article WHERE url_canonical = ? LIMIT 1', (url_canonical,)
        ).fetchone()
        is not None
    )


def _store_page(
    conn: sqlite3.Connection,
    oid: int,
    source_feed: str,
    result: tuple[int, str, str],
) -> bool:
    status, html, final_url = result
    canon = _canonic_url(final_url)
    content_hash = hashlib.sha256(html.encode('utf-8', 'replace')).hexdigest()
    now = datetime.now(UTC).isoformat()

    try:
        with conn:
            cur = conn.execute(
                'INSERT INTO article '
                '(outlet_id, url, url_canonical, source_feed, scraped_date) '
                'VALUES (?, ?, ?, ?, ?)',
                (oid, final_url, canon, source_feed, now),
            )
            conn.execute(
                'INSERT INTO raw_page '
                '(article_id, raw_html, http_status, content_hash, fetched_date) '
                'VALUES (?, ?, ?, ?, ?)',
                (cur.lastrowid, html, status, content_hash, now),
            )
    except sqlite3.IntegrityError:
        return False
    return True


def _process_url(
    conn: sqlite3.Connection,
    session: requests.Session,
    raw_url: str,
    oid: int,
    source_feed: str,
) -> str:
    if _already_have(conn, _canonic_url(raw_url)):
        return 'skipped'
    result = _fetch(session, raw_url)
    if result is None:
        return 'failed'
    return 'stored' if _store_page(conn, oid, source_feed, result) else 'skipped'
