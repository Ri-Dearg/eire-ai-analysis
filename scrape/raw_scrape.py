"""Use a list of URLs to fetch and store raw HTML pages in the DB, with metadata.

Designed to be used with URLs from sitemaps, but can be used with any list of article URLs.
Optionally respects robots.txt and retries transient failures with pause.
Logs progress and outcomes.
"""

import hashlib
import logging
import random
import sqlite3
import time
from collections import Counter
from datetime import UTC, datetime
from urllib import robotparser  # add to the urllib imports
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

# LOGGING
logger = logging.getLogger(__name__)


# DATABASE
# Path to DB file.
DB_PATH = './data/dataset.db'

# SCRAPER
# User agent string for scraper.
USER_AGENT = (
    'CapstoneResearchBot/0.1 (HDip Data Analytics, DBS; contact: 20074605@mydbs.ie)'
)
# Request timeout in seconds.
REQUEST_TIMEOUT = 20
# Polite delay range between requests in seconds (min, max).
DELAY_RANGE = (4.0, 6.0)

# CANONICALISATION
# Query parameters to drop during URL canonicalisation.
_TRACKING_PREFIXES = ('utm_',)
_TRACKING_KEYS = {'fbclid', 'gclid', 'mc_cid', 'mc_eid'}

# RETRIES
# Retry transient failures: transport errors + these HTTP statuses.
MAX_RETRIES = 3
RETRY_STATUSES = {429, 500, 502, 503, 504}
PAUSE_BASE = 4.0
PAUSE_CAP = 60.0
RESPECT_ROBOTS = True
ACCEPTED_ERRORS = 400


# ---------- ARTICLE PROCESSING ----------


def _already_have(conn: sqlite3.Connection, url_canonical: str) -> bool:
    """Check if the URL has already been parsed and stored.

    Args:
        conn (sqlite3.Connection): Connection to DB.
        url_canonical (str): Canonicalised URL.

    Returns:
        bool: Is stored in the DB?

    """
    return (
        conn.execute(
            'SELECT 1 FROM article WHERE url_canonical = ? LIMIT 1', (url_canonical,)
        ).fetchone()
        is not None
    )


def _canonic_url(url: str) -> str:
    """Strip trackers from end of URL.

    Args:
        url (str): URL to strip.

    Returns:
        str: Canonicalised URL.

    """
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


def _fetch(session: requests.Session, url: str) -> tuple[int, str, str] | None:
    """Collect and return article data.

    Args:
        session (requests.Session): User-agent session.
        url (str): Url to parse.

    Returns:
        tuple[int, str, str] | None: critical response data.

    """
    # Loop with retries for transport errors and certain HTTP statuses.
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                logger.exception('  ! transport error %s:', url)
                return None
            # Wait with exponential pause and jitter before retrying.
            wait = _pause(attempt)
            logger.warning('retry %d/%d in %.1fs (%s)', attempt + 1, MAX_RETRIES, wait, e)
            time.sleep(wait)
            continue

        # Retry on certain HTTP statuses.
        if resp.status_code in RETRY_STATUSES and attempt < MAX_RETRIES:
            wait = _retry_after(resp) or _pause(attempt)
            logger.warning(
                '%s %s; retry %d/%d in %.1fs',
                resp.status_code,
                url,
                attempt + 1,
                MAX_RETRIES,
                wait,
            )
            # Wait before retrying.
            time.sleep(wait)
            continue

        # Return critical data for successful response or error.
        return resp.status_code, resp.text, resp.url
    return None


def _pause(attempt: int) -> float:
    """Calculate time to pause.

    Args:
        attempt (int): Number of attempts.

    Returns:
        float: Length of pause.

    """
    return random.uniform(0, min(PAUSE_BASE * (2**attempt), PAUSE_CAP))


def _process_url(
    conn: sqlite3.Connection,
    session: requests.Session,
    raw_url: str,
    oid: int,
    source_feed: str,
) -> str:
    """Check if the URL is already stored, skip if so; otherwise fetch and store it.

    Args:
        conn (sqlite3.Connection): DB connection.
        session (requests.Session): User-agent session for fetching.
        raw_url (str): Url from sitemap.
        oid (int): Outlet ID.
        source_feed (str): Source feed type.

    Returns:
        str: value of result: 'skipped', 'failed', 'stored', or 'not_stored'.

    """
    if _already_have(conn, _canonic_url(raw_url)):
        return 'skipped'
    result = _fetch(session, raw_url)
    if result is None:
        return 'failed'
    return 'stored' if _store_page(conn, oid, source_feed, result) else 'not_stored'


def _retry_after(resp: requests.Response) -> float | None:
    """Get retry delay from Retry-After header.

    Args:
        resp (requests.Response): Response object.

    Returns:
        float | None: Retry delay in seconds.

    """
    value = resp.headers.get('Retry-After')
    return min(float(value), PAUSE_CAP) if value and value.isdigit() else None


def _store_page(
    conn: sqlite3.Connection,
    oid: int,
    source_feed: str,
    result: tuple[int, str, str],
) -> bool:
    """Store and insert data into DB.

    Args:
        conn (sqlite3.Connection): Connection to DB.
        oid (int): Outlet ID.
        source_feed (str): Source feed type.
        result (tuple[int, str, str]): Fetched data.

    Returns:
        bool: Was the data stored successfully?

    """
    status, html, final_url = result
    canon = _canonic_url(final_url)
    content_hash = hashlib.sha256(html.encode('utf-8', 'replace')).hexdigest()
    now = datetime.now(UTC).isoformat()

    try:
        with conn:
            # Attempt to insert article and raw page data.
            # If the canonical URL already exists, skip storing.
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
    except sqlite3.IntegrityError as err:
        if 'UNIQUE constraint failed: article.url_canonical' in str(err):
            return False
        raise
    return True


# ---------- ROBOTS ----------
def _host_base(url: str) -> str:
    """Split url to get host base url.

    Args:
        url (str): website url.

    Returns:
        str: base url for host.

    """
    parts = urlsplit(url)
    return f'{parts.scheme}://{parts.netloc}'


def _robots(session: requests.Session, base: str) -> robotparser.RobotFileParser:
    """Parse site robots.txt.

    Args:
        session (requests.Session): opened user-agent session.
        base (str): base url for website.

    Returns:
        robotparser.RobotFileParser: Parsed robot.txt file.

    """
    robot_parser = robotparser.RobotFileParser()
    try:
        resp = session.get(f'{base}/robots.txt', timeout=REQUEST_TIMEOUT)
        robot_parser.parse(
            resp.text.splitlines() if resp.status_code < ACCEPTED_ERRORS else []
        )
    except requests.RequestException:
        robot_parser.parse([])  # fail-open if robots.txt can't be read
    return robot_parser


# ---------- SETUP ----------
def _create_session() -> requests.Session:
    """Create a session and user agent for crawling.

    Returns:
        requests.Session: Session for crawling.

    """
    session = requests.Session()
    session.headers.update({'User-Agent': USER_AGENT})
    return session


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
        conn.execute('PRAGMA quick_check')
        logger.info('Connected to %s', db_path)
    except sqlite3.OperationalError:
        logger.exception('Failed to connect to %s', db_path)
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
        err_msg = 'outlet {name!r} not found in DB.'
        raise ValueError(err_msg.format(name=name))
    return row[0]


# ---------- MAIN FUNCTION ----------
def ingest(
    urls: list[str],
    outlet_name: str,
    source_feed: str,
    db_path: str = DB_PATH,
    delay_range: tuple[float, float] = DELAY_RANGE,
) -> dict:
    conn = _connect(db_path)
    oid = _outlet_id(conn, outlet_name)

    with _create_session() as session:
        counts = Counter(
            {'stored': 0, 'skipped': 0, 'blocked': 0, 'failed': 0, 'not_stored': 0}
        )
        rules = _robots(session, _host_base(urls[0])) if RESPECT_ROBOTS and urls else None

        try:
            for raw_url in urls:
                if rules is not None and not rules.can_fetch(USER_AGENT, raw_url):
                    counts['blocked'] += 1
                    logger.info('blocked: %s', raw_url)
                    continue
                outcome = _process_url(conn, session, raw_url, oid, source_feed)
                counts[outcome] += 1
                logger.info('%s: %s', outcome, raw_url)
                if outcome != 'skipped':
                    time.sleep(random.uniform(*delay_range))
        finally:
            conn.close()

        logger.info(
            'done: %d stored, %d skipped, %d blocked, %d failed, %d not stored',
            counts['stored'],
            counts['skipped'],
            counts['blocked'],
            counts['failed'],
            counts['not_stored'],
        )
        return dict(counts)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )
    sample = [
        'https://www.rte.ie/news/business/2026/0331/1566027-unilever-nears-deal-to-merge-foods-unit-with-mccormick/',
        'https://www.rte.ie/news/business/2026/0331/1566035-business-post-appoints-mark-beard-as-ceo/',
        'https://www.rte.ie/news/business/2026/0331/1566030-heathrow-airports-fees-set-to-rise-by-1/',
        'https://www.rte.ie/news/ireland/2026/0330/1565830-my-lovely-horse-rescue/',
        'https://www.rte.ie/news/world/2026/0330/1565833-cuba-russia-us/',
        'https://www.rte.ie/news/business/2026/0331/1566020-mortgage-approvals-reached-almost-12-billion-in-february/',
        'https://www.rte.ie/brainstorm/2026/0331/1566019-iran-strait-of-harmuz-defence-strategy-persian-gulf/',
        'https://www.rte.ie/news/munster/2026/0330/1565972-cock-fighting/',
        'https://www.rte.ie/news/ulster/2026/0331/1566011-marian-beattie-appeal/',
        'https://www.rte.ie/news/ulster/2026/0330/1565902-schwarzenegger-belfast-honour/',
        'https://www.rte.ie/brainstorm/2026/0324/1565008-early-education-creche-children-education-care-problems/',
        'https://www.rte.ie/news/ireland/2026/0330/1565866-antoin-duffy-court/',
        'https://www.rte.ie/news/dublin/2026/0331/1565981-dublin-property-prices/',
        'https://www.rte.ie/news/middle-east/2026/0330/1565832-iran-war/',
        'https://www.rte.ie/news/newslens/2026/0330/1566004-air-canada-ceo/',
    ]
    ingest(sample, outlet_name='rte', source_feed='sitemap')
