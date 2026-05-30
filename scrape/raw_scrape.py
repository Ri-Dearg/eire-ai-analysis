"""WIP."""

import sqlite3

import requests

DB_PATH = './data/dataset.db'

USER_AGENT = (
    'CapstoneResearchBot/0.1 (HDip Data Analytics, DBS; contact: 20074605@mydbs.ie)'
)

REQUEST_TIMEOUT = 20

DELAY_RANGE = (4.0, 6.0)

_TRACKING_PREFIXES = ('utm_',)
_TRACKING_KEYS = {'fbclid', 'gclid', 'mc_cid', 'mc_eid'}


def connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Open a connection with foreign keys enforced.

    Args:
        db_path (str, optional): DB location. Defaults to DB_PATH.

    Returns:
        sqlite3.Connection: Connection to DB.

    """
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
    row = conn.execute('SELECT id FROM outlet WHERE name = ?', (name,)).fetchone()
    if row is None:
        print(f'outlet {name!r} not found in DB.')
        raise ValueError
    return row[0]


def create_session():
    session = requests.Session()
    session.headers.update({'User-Agent': USER_AGENT})
    return session


def fetch(session: requests.Session, url: str):
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        print(f'  ! transport error {url}: {e}')
        return None
    return resp.status_code, resp.text, resp.url
