"""WIP."""

import sqlite3
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


def canonic_url(url: str) -> str:
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


def connect(db_path: str = DB_PATH) -> sqlite3.Connection:
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


def outlet_id(conn: sqlite3.Connection, name: str) -> int:
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


def create_session():
    # Create a requests session with a custom user agent.
    session = requests.Session()
    session.headers.update({'User-Agent': USER_AGENT})
    return session


def fetch(session: requests.Session, url: str):
    # Fetch a URL with error handling and timeout.
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        print(f'  ! transport error {url}: {e}')
        return None
    return resp.status_code, resp.text, resp.url
