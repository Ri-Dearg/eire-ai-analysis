"""Parse the articles offline to extract the article body.

Provides a fix for boilerplate-tail stripping.

Two body strings are produced per article and kept side by side on purpose:

body_raw, the exact dry-run extraction and body_text, the same content with
template  stripped. This is the detector-facing text.

The DB is read strictly read-only; raw_page is never altered. Output is a
single CSV with one row per article and no drop decisions,so the body pass runs
once and the easier curation can be re-run freely.
"""

from __future__ import annotations

import csv
import hashlib
import html as ihtml
import json
import logging
import os
import re
import sqlite3
import sys
import time

# Multiprocessing structure suggested and designed by AI for faster parsing.
from multiprocessing import Pool
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------- DIRECTORIES ----------
ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / 'data' / 'dataset.db'
OUT_DIR = ROOT / 'data'
OUT_CSV = OUT_DIR / 'parsed_all.csv'

# ---------- RUNNER VARIABLES ----------
HTTP_OK = 200
N_WORKERS = 4
BATCH = 200
TIME_BUDGET = float(os.environ.get('PARSE_BUDGET', '0'))

# ---------- DATE VARIABLES ----------
RELEASE = '2022-11-30'
TODAY = '2026-06-09'  # frozen corpus snapshot; articles after this are out of scope
MONTHS = [
    'January',
    'February',
    'March',
    'April',
    'May',
    'June',
    'July',
    'August',
    'September',
    'October',
    'November',
    'December',
]
MONTH_NUM = {month: i + 1 for i, month in enumerate(MONTHS)}

# ---------- ARTICLE VARIABLES ----------
COLS = [
    'article_id',
    'outlet',
    'url',
    'url_canonical',
    'http_status',
    'published_date',
    'date_src',
    'period',
    'author',
    'section',
    'is_wire',
    'wire_match',
    'is_otd',
    'sub_excl',
    'gript_premium',
    'body_len_raw',
    'body_sha1',
    'word_count',
    'body_text',
    'parse_error',
]

# ---------- REGEX VARIABLES ----------
DOTALL_I = re.DOTALL | re.IGNORECASE

EXAMINER_TAIL_RE = re.compile(
    r'\s*(Try from only|CONNECT WITH US TODAY|Be the first to know|'
    r'Sign up for our|Already a subscriber\?\s*Sign in|'
    r'More in this section).*$',
    DOTALL_I,
)

LIB_DONATION_RE = re.compile(
    r'TheLiberal\.ie\s+won.{0,3}t\s+quit\s+Please support us with a small '
    r'donation on PayPal!?\s*(?:Donate Now)?\s*',
    re.IGNORECASE,
)
LIB_END_RE = re.compile(
    r'\s*(Tell us your thoughts in the|Please signup free to our newsletter|'
    r'About Us Advertise|Copyright\s+20\d\d\s+TheLiberal|'
    r'Privacy settings Close this module).*$',
    DOTALL_I,
)
LIB_ENTRY_RE = re.compile(
    r'<div[^>]*class=["\'][^"\']*entry-content[^"\']*["\'][^>]*>(.*)', DOTALL_I
)
LIB_IMGSRC_RE = re.compile(r'^Image source:\s*\S+\s*', re.IGNORECASE)

MONTHS_RE = '|'.join(MONTHS)


RTE_CONSENT_RE = re.compile(r'\s*We need your consent to load.*$', DOTALL_I)
RTE_ROLE_RE = re.compile(
    r'(?:[A-Z][\w’\'&.-]*\s+){0,5}'  # noqa: RUF001
    r'(?:Correspondent|Correspondents|Editor|Reporter|Analyst|Desk)\s+'
)

WIRE_RE = re.compile(
    r'\b(Reuters|Associated Press|AP\b|Agence France|AFP|Press Association|'
    r'PA Media|Additional reporting by|\bPA\b)'
)


# ---------- HTML PARSING ----------
def best_article_body(stripped_html: str) -> str:
    """Return the longest <article> block's <p> text (else whole page).

    Args:
        stripped_html (str): HTML stripped of style/script blocks.

    Returns:
        str: paragraph text blocks.

    """
    blocks = re.findall(r'<article\b[^>]*>(.*?)</article>', stripped_html, flags=DOTALL_I)
    if not blocks:
        return p_text(stripped_html)
    return max((p_text(block) for block in blocks), key=len)


def _detag(segment: str) -> str:
    """Strip HTML tags, unescape entities, and collapse whitespace to one line.

    Args:
        segment (str): segment of text to strip tags from

    Returns:
        str: Segment without tags.

    """
    return re.sub(r'\s+', ' ', ihtml.unescape(re.sub(r'<[^>]+>', ' ', segment))).strip()


def flat(data: object) -> list[str]:
    """Flatten nested lists to a list of strings.

    Args:
        data (object): nested lists or dicts.

    Returns:
        list[str]: Simple list of strings.

    """
    out: list[str] = []
    stack = [data]
    while stack:
        x = stack.pop()
        if isinstance(x, list):
            stack.extend(x)
        elif x:
            out.append(str(x))
    return out[::-1]


def iso_from_dt(string: str | None) -> str:
    """Return the leading YYYY-MM-DD of a datetime string, or ''.

    Args:
        string (str | None): datetime string

    Returns:
        str: iso datetime as string

    """
    if not string:
        return ''
    date_match = re.match(r'(\d{4})-(\d{2})-(\d{2})', string)
    return date_match.group(0) if date_match else ''


def jsonld_nodes(html: str) -> list[dict]:
    """Return all JSON nodes in html.

    Parses with strict=False because Examiner blocks carry literal
    control characters that fail strict JSON.

    Args:
        html (str): HTML made up of JsonDL nodes.

    Returns:
        list[dict]: A list of JsonDL nodes as dicts.

    """
    nodes: list[dict] = []
    for block in re.findall(
        r'<script[^>]*application/ld\+json[^>]*>(.*?)</script[^>]*>', html, flags=DOTALL_I
    ):
        try:
            data = json.loads(block.strip(), strict=False)
        except (ValueError, TypeError):
            continue
        stack = [data]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                if '@graph' in item:
                    stack.extend(item['@graph'])
                else:
                    nodes.append(item)
            elif isinstance(item, list):
                stack.extend(item)
    return nodes


def p_text(segment: str) -> str:
    """Return concatenated visible text of the <p> tags in segment.

    Args:
        segment (str): String segment with p tags.

    Returns:
        str: Text without p tags.

    """
    ps = re.findall(r'<p\b[^>]*>(.*?)</p>', segment, flags=DOTALL_I)
    return _detag(' '.join(ps))


# TODO(Rory): Simplify this
def period_of(date_iso: str) -> str:  # noqa: PLR0911
    """Map an ISO date to a corpus period label.

    Returns one of out_lo (<2019), pre (2019 .. pre-release),
    straddle (release .. 2022-12-31), mid (2023-24), post (2025-26),
    out_hi (after the snapshot), or '' for a missing date.

    Args:
        date_iso (str): ISO format datetime.

    Returns:
        str: pre or post ChatGPT period.

    """
    if not date_iso:
        return ''
    if date_iso < '2019-01-01':
        return 'out_lo'
    if date_iso < RELEASE:
        return 'pre'
    if date_iso <= '2022-12-31':
        return 'straddle'
    if date_iso <= '2024-12-31':
        return 'mid'
    if date_iso <= TODAY:
        return 'post'
    return 'out_hi'


def strip_style_scripts(html: str) -> str:
    """Return html with <script> and <style> blocks removed.

    Args:
        html (str): html

    Returns:
        str: Html with style and script stripped.

    """
    html = re.sub(r'<script\b[^>]*>.*?</script[^>]*>', ' ', html, flags=DOTALL_I)
    return re.sub(r'<style\b[^>]*>.*?</style[^>]*>', ' ', html, flags=DOTALL_I)


def unescape(string: object) -> str:
    """HTML-unescape and strip; tolerate None.

    Args:
        string (object): String to unescape.

    Returns:
        str: String with items unescaped.

    """
    return ihtml.unescape(str(string or '')).strip()


# ---------- CONTENT PARSING ----------
def author_name(article: object) -> str | None:
    """Resolve a JSON-LD author value (dict/list/str) to a name.

    Args:
        article (object): Article object.

    Returns:
        str | None: Author's name.

    """
    if isinstance(article, dict):
        return article.get('name')
    if isinstance(article, list) and article:
        return author_name(article[0])
    return article if isinstance(article, str) else None


def _features(  # noqa: PLR0913
    *,
    body_raw: str,
    body_text: str,
    date_iso: str,
    date_src: str = '',
    author: str = '',
    section: str = '',
    is_wire: int = 0,
    wire_match: str = '',
    sub_excl: int = 0,
    gript_premium: int = 0,
    is_otd: int = 0,
) -> dict:
    """Assemble the feature dict consumed by _row_record.

    Centralises the feature returns, so each extractor sets only the fields that differ.

    Args:
        body_raw (str): Raw article body
        body_text (str): Text extracted from the body.
        date_iso (str): ISO date format
        date_src (str, optional): source of the date. Defaults to ''.
        author (str, optional): Name of author. Defaults to ''.
        section (str, optional): Section of the article. Defaults to ''.
        is_wire (int, optional): Flag for a wire service. Defaults to 0.
        wire_match (str, optional): Text matching the wire service. Defaults to ''.
        sub_excl (int, optional): Flag for exclusive content. Defaults to 0.
        gript_premium (int, optional): Flag for premium content. Defaults to 0.
        is_otd (int, optional): Flag for OTD content. Defaults to 0.

    Returns:
        dict: Dictionary of features to be stored.

    """
    return {
        'author': author,
        'date_iso': date_iso,
        'date_src': date_src,
        'section': section,
        'body_raw': body_raw,
        'body_text': body_text,
        'is_wire': is_wire,
        'wire_match': wire_match,
        'sub_excl': sub_excl,
        'gript_premium': gript_premium,
        'is_otd': is_otd,
    }


def _ld_author_date(html: str) -> tuple[str, str, str]:
    """Return info from the NewsArticle JSON-LD node.

    Shared by the RTE and Examiner extractors: reads the author and
    datePublished, falling back to the article:published_time meta tag
    when JSON-LD carries no date.

    Args:
        html (str): HTML to parse.

    Returns:
        tuple[str, str, str]: author, date_iso, date_src

    """
    nodes = jsonld_nodes(html)
    article = next(
        (node for node in nodes if 'NewsArticle' in str(node.get('@type', ''))), {}
    )
    author = unescape(author_name(article.get('author')) or '')
    date_iso = iso_from_dt(article.get('datePublished') or '')
    date_src = 'ld' if date_iso else ''
    if not date_iso:
        date_iso = iso_from_dt(meta_content(html, 'article:published_time'))
        date_src = 'meta' if date_iso else ''
    return author, date_iso, date_src


def meta_content(html: str, name: str) -> str:
    """Return the content of the <meta> whose name/property is name.

    Args:
        html (str): HTML content.
        name (str): Name of meta content.

    Returns:
        str: Meta Content parsed.

    """
    esc = re.escape(name)
    meta_match = re.search(
        rf'<meta[^>]*(?:name|property)=["\']{esc}["\'][^>]*content=["\']([^"\']*)',
        html,
        flags=re.IGNORECASE,
    )
    if not meta_match:
        meta_match = re.search(
            rf'<meta[^>]*content=["\']([^"\']*)["\'][^>]*(?:name|property)='
            rf'["\']{esc}["\']',
            html,
            flags=re.IGNORECASE,
        )
    return meta_match.group(1) if meta_match else ''


def _wire(body_raw: str) -> tuple[int, str]:
    """Return (is_wire, matched_text) from scanning body_raw for wire credits.

    Args:
        body_raw (str): Raw article content.

    Returns:
        tuple[int, str]: 0 or 1 based on whether it is a wire or not, the wire match.

    """
    match = WIRE_RE.search(body_raw)
    return int(bool(match)), (match.group(0) if match else '')


# ---------- OUTLET SPECIFIC PARSING ----------
def _strip_examiner(body_raw: str) -> str:
    """Drop the Examiner subscription-promo / newsletter / read-more.

    Args:
        body_raw (str): Raw article body.

    Returns:
        str: Raw body without examiner tail.

    """
    return EXAMINER_TAIL_RE.sub('', body_raw).strip()


def _strip_liberal(entry_html: str) -> str:
    """Return the Liberal article prose.

    Extracts <p> prose from the entry-content region
    removes the inline donation widget, and trims the leading Image source.

    Args:
        entry_html (str): HTML at the start of the page.

    Returns:
        str: Entry without the extra prose content.

    """
    b = p_text(entry_html)
    b = LIB_DONATION_RE.sub(' ', b)
    b = LIB_IMGSRC_RE.sub('', b)
    b = LIB_END_RE.sub('', b)
    return re.sub(r'\s+', ' ', b).strip()


def _strip_rte(body_raw: str, author: str) -> str:
    """Drop RTE's consent widget and leading block.

    Args:
        body_raw (str): Raw article body
        author (str): Author name.

    Returns:
        str: Article without widget.

    """
    block = RTE_CONSENT_RE.sub('', body_raw).strip()
    if author and block.startswith(author):
        block = block[len(author) :].lstrip()
        rte_match = RTE_ROLE_RE.match(block)
        if rte_match:
            block = block[rte_match.end() :]
    return block.strip()


def feat_examiner(html: str, url: str) -> dict:
    """Extract Examiner fields; section (IE-<word>/ stripped)/URL.

    Args:
        html (str): Article html.
        url (str): url for the article.

    Returns:
        dict: Dict of values for the DB row.

    """
    author, date_iso, date_src = _ld_author_date(html)

    section_meta = (
        re.sub(
            r'^IE-[a-z]+/', '', meta_content(html, 'article:section'), flags=re.IGNORECASE
        )
        .strip()
        .lower()
    )
    section_match = re.search(r'irishexaminer\.com/([^/]+)/', url)
    section = section_meta or (section_match.group(1) if section_match else '')
    low = html.lower()
    body_raw = best_article_body(strip_style_scripts(html))
    is_wire, wire_match = _wire(body_raw)

    return _features(
        author=author,
        date_iso=date_iso,
        date_src=date_src,
        section=section,
        body_raw=body_raw,
        body_text=_strip_examiner(body_raw),
        is_wire=is_wire,
        wire_match=wire_match,
        sub_excl=int('exclusive subscriber content' in low),
    )


def feat_gript(html: str, url: str) -> dict:  # noqa: ARG001
    """Extract Gript fields from the stored WordPress REST JSON.

    Args:
        html (str): HTML of article.
        url (str): URL of article.

    Returns:
        dict: Features for article rows.

    """
    data = json.loads(html)
    content = data.get('content')
    content: Any = (
        content.get('rendered') if isinstance(content, dict) else (content or '')
    )
    body_raw = _detag(content)
    date_iso = iso_from_dt(data.get('date_gmt') or '')
    date_src = 'gmt' if date_iso else ''
    if not date_iso:
        date_iso = iso_from_dt(data.get('date') or '')
        date_src = 'date' if date_iso else ''
    yoast_head = data.get('yoast_head_json') or {}
    author = unescape(
        (yoast_head.get('author') if isinstance(yoast_head, dict) else '') or ''
    )
    title = data.get('title')
    title = title.get('rendered') if isinstance(title, dict) else (title or '')
    title = unescape(re.sub(r'<[^>]+>', '', str(title)))
    sections: list[str] = []
    schema = yoast_head.get('schema') if isinstance(yoast_head, dict) else None
    if isinstance(schema, dict):
        for node in schema.get('@graph', []):
            if isinstance(node, dict) and node.get('articleSection'):
                sections.extend(flat(node['articleSection']))
    section = '|'.join(dict.fromkeys(sections))
    lower_case = content.lower()
    title_upper = title.upper()
    is_otd = int(
        'on this day' in section.lower()
        or title_upper.startswith(('OTD:', 'ON THIS DAY'))
    )

    return _features(
        author=author,
        date_iso=date_iso,
        date_src=date_src,
        section=section,
        body_raw=body_raw,
        body_text=body_raw,
        gript_premium=int(
            'memberful-global-teaser-content' in lower_case
            or 'premium' in section.lower()
        ),
        is_otd=is_otd,
    )


def feat_liberal(html: str, url: str) -> dict:  # noqa: ARG001
    """Extract Liberal fields from the header byline and entry-content.

    Args:
        html (str): HTML of article.
        url (str): URL of article.

    Returns:
        dict: Features for article rows.

    """
    html_clean = strip_style_scripts(html)
    am = re.search(
        r'<a[^>]*class=["\'][^"\']*\bfn\b[^"\']*["\'][^>]*>(.*?)</a>',
        html_clean,
        flags=DOTALL_I,
    ) or re.search(
        r'<a[^>]*href=["\'][^"\']*/author/[^"\']*["\'][^>]*>(.*?)</a>',
        html_clean,
        flags=DOTALL_I,
    )
    author = unescape(re.sub(r'<[^>]+>', '', am.group(1)) if am else '')
    content_match = re.search(r'class=["\'][^"\']*post-\d+[^"\']*["\']', html_clean)
    section = (
        '|'.join(re.findall(r'category-([a-z0-9-]+)', content_match.group(0)))
        if content_match
        else ''
    )
    date_iso = ''
    heading_match = re.search(
        r'<h1[^>]*entry-title[^>]*>.*?</h1>', html_clean, flags=DOTALL_I
    ) or re.search(r'<h1[^>]*>.*?</h1>', html_clean, flags=DOTALL_I)
    if heading_match:
        win = re.sub(
            r'<[^>]+>', ' ', html_clean[heading_match.end() : heading_match.end() + 700]
        )
        date_match = re.search(rf'({MONTHS_RE})\s+(\d{{1,2}}),\s+(\d{{4}})', win)
        if date_match:
            date_iso = (
                f'{int(date_match.group(3)):04d}-{MONTH_NUM[date_match.group(1)]:02d}'
                f'-{int(date_match.group(2)):02d}'
            )
    entry_match = LIB_ENTRY_RE.search(html_clean)
    entry = entry_match.group(1) if entry_match else ''
    body_raw = _detag(entry)
    is_wire, wire_match = _wire(body_raw)

    return _features(
        author=author,
        date_iso=date_iso,
        date_src='header' if date_iso else '',
        section=section,
        body_raw=body_raw,
        body_text=_strip_liberal(entry),
        is_wire=is_wire,
        wire_match=wire_match,
    )


def feat_rte(html: str, url: str) -> dict:
    """Extract RTE fields; date falls back to URL path for live pages.

    Args:
        html (str): Article html.
        url (str): url for the article.

    Returns:
        dict: Dict of values for the DB row.

    """
    author, date_iso, date_src = _ld_author_date(html)
    if not date_iso:
        date_match = re.search(r'/news/(?:[^/]+/)?(\d{4})/(\d{2})(\d{2})/', url)
        if date_match:
            date_iso = '{}-{}-{}'.format(*date_match.groups())
            date_src = 'url'

    segment_match = re.search(r'rte\.ie/news/([^/]+)/', url)
    segment = segment_match.group(1) if segment_match else ''
    section = '' if re.fullmatch(r'\d{4}', segment) else segment
    body_raw = best_article_body(strip_style_scripts(html))
    is_wire, wire_match = _wire(body_raw)

    return _features(
        author=author,
        date_iso=date_iso,
        date_src=date_src,
        section=section or 'news',
        body_raw=body_raw,
        body_text=_strip_rte(body_raw, author),
        is_wire=is_wire,
        wire_match=wire_match,
    )


EXTRACT = {
    'gript': feat_gript,
    'rte': feat_rte,
    'irish_examiner': feat_examiner,
    'the_liberal': feat_liberal,
}


# ---------- DATABASE RECORDING ----------
def _row_record(row: tuple) -> dict:
    """Build one record from a (id, outlet, url, curl, status, raw) row.

    Args:
        row (tuple): Features of one row.

    Returns:
        dict: Complete record to output.

    """
    aid, outlet, aurl, curl, status, raw = row
    record: dict[str, str | int] = dict.fromkeys(COLS, '')
    record.update(
        article_id=aid, outlet=outlet, url=aurl, url_canonical=curl, http_status=status
    )
    if not raw:
        record['parse_error'] = 'no_raw'
        return record
    if status != HTTP_OK:
        record['parse_error'] = f'http_{status}'
        return record
    try:
        features = EXTRACT[outlet](raw, curl or aurl)
    except Exception as exc:  # noqa: BLE001
        record['parse_error'] = type(exc).__name__
        return record
    body_raw = features['body_raw']
    body_text = features['body_text']
    norm = re.sub(r'\s+', ' ', body_raw).strip().lower()
    record.update(
        published_date=features['date_iso'],
        date_src=features['date_src'],
        period=period_of(features['date_iso']),
        author=features['author'],
        section=features['section'],
        is_wire=features['is_wire'],
        wire_match=features['wire_match'],
        is_otd=features['is_otd'],
        sub_excl=features['sub_excl'],
        gript_premium=features['gript_premium'],
        body_len_raw=len(body_raw),
        body_sha1=hashlib.sha1(norm.encode()).hexdigest() if norm else '',  # noqa: S324
        word_count=len(body_text.split()),
        body_text=body_text,
    )
    return record


def worker(worker_id: int) -> int:
    """Parse the id-stripe id %% N_WORKERS == worker_id into part_<worker_id>.csv.

    Args:
        worker_id (int): ID number of running worker.

    Returns:
        int: Remaining parts.

    """
    connection = sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=60)
    cursor = connection.cursor()
    ids = [
        row[0]
        for row in cursor.execute(
            f'SELECT id FROM article WHERE (id % {N_WORKERS})={worker_id} ORDER BY id'  # noqa: S608
        )
    ]

    path = OUT_DIR / f'part_{worker_id}.csv'
    last_done = 0
    if path.exists():
        with path.open(encoding='utf-8') as part:
            for line in part:
                head = line.split(',', 1)[0]
                if head.isdigit():
                    last_done = max(last_done, int(head))

    new_file = last_done == 0
    ids = [id_num for id_num in ids if id_num > last_done]
    current_time = time.time()
    num = 0
    done = True

    with path.open('a', newline='', encoding='utf-8') as fh:
        write = csv.DictWriter(fh, fieldnames=COLS)
        if worker_id == 0 and new_file:
            write.writeheader()
        for i in range(0, len(ids), BATCH):
            if TIME_BUDGET and time.time() - current_time > TIME_BUDGET:
                done = False
                break
            chunk = ids[i : i + BATCH]
            placeholders = ','.join('?' * len(chunk))
            query = (
                'SELECT a.id, o.name, a.url, a.url_canonical, r.http_status, '  # noqa: S608
                'r.raw_html FROM article a JOIN outlet o ON o.id=a.outlet_id '
                'LEFT JOIN raw_page r ON r.article_id=a.id '
                f'WHERE a.id IN ({placeholders})'
            )
            for db_row in cursor.execute(query, chunk):
                write.writerow(_row_record(db_row))
                num += 1
            fh.flush()

    connection.close()
    remaining = 0 if done else len(ids) - num
    logger.info(
        'w%s wrote %s, remaining %s, %s}s',
        worker_id,
        num,
        remaining,
        time.time() - current_time,
    )
    return remaining


def merge_parts() -> None:
    """Concatenate the part files into OUT_CSV."""
    with OUT_CSV.open('w', encoding='utf-8') as output:
        for worker_id in range(N_WORKERS):
            output.write((OUT_DIR / f'part_{worker_id}.csv').read_text(encoding='utf-8'))


def main() -> int:
    """Run the parse pass across all workers and merge to OUT_CSV.

    Returns:
        int: Complete or complete run.

    """
    current_time = time.time()

    with Pool(N_WORKERS) as pool:
        remaining = pool.map(worker, range(N_WORKERS))
    total_remaining = sum(remaining)

    if total_remaining:
        logger.info(
            'PASS incomplete, remaining=%s, %s}s',
            total_remaining,
            time.time() - current_time,
        )
        return 1

    merge_parts()
    logger.info('PARSE DONE -> %s in %s}s', OUT_CSV, time.time() - current_time)
    return 0


if __name__ == '__main__':
    sys.exit(main())
