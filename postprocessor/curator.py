from __future__ import annotations

import csv
import hashlib
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------- DIRECTORIES ----------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
PARSED = DATA / 'parsed_all.csv'

OUTLETS = ('gript', 'irish_examiner', 'rte', 'the_liberal')

PRE_FINE = ('pre',)
POST_FINE = ('straddle', 'mid', 'post')
DROP_OOR = ('out_lo', 'out_hi', '')

MIN_BODY_CHARS = 400
STUB_MIN_WORDS = 20
NONPROSE_MIN_WORDS = 120
NONPROSE_MAX_WPS = 90

_TERM_RE = re.compile(r'[.!?…]')
_WS_RE = re.compile(r'\s+')


def _is_nonprose(body: str, word_count: int) -> bool:
    """Return True when a body reads as non-prose (a table or unstructured dump).

    Args:
        body (str): The article body text.
        word_count (int): Pre-computed word count of ``body``.

    Returns:
        bool: True if the body should be dropped as non-prose.

    """
    terminators = len(_TERM_RE.findall(body))
    if terminators == 0:
        return True
    return (
        word_count >= NONPROSE_MIN_WORDS and word_count / terminators > NONPROSE_MAX_WPS
    )


def _norm_hash(body: str) -> str:
    """Return a hash of the body normalised for near-duplicate detection.

    Args:
        body (str): The article body text.

    Returns:
        str: SHA1 hex digest of the lower-cased, whitespace-collapsed body.

    """
    norm = _WS_RE.sub(' ', body.lower()).strip()
    return hashlib.sha1(norm.encode('utf-8')).hexdigest()  # noqa: S324 Not a security function


def drop_reason(row: dict, seen: set[str], seen_norm: set[str]) -> str:  # noqa: PLR0911
    """Return the first applicable drop reason for a row, or '' if it survives.

    Args:
        row (dict): The article row.
        seen (set[str]): Raw body sha1 hashes already kept.
        seen_norm (set[str]): Normalised body hashes already kept.

    Returns:
        str: The drop reason, or '' if the row is kept.

    """
    if row['parse_error'] or row['http_status'] != '200':
        return 'non200_or_noraw'
    if int(row['body_len_raw'] or 0) < MIN_BODY_CHARS:
        return 'thin_lt400'
    body = row['body_text']
    if not body.strip():
        return 'empty_clean_body'
    word_count = len(body.split())
    if word_count < STUB_MIN_WORDS:
        return 'stub'
    if _is_nonprose(body, word_count):
        return 'nonprose'
    if row['outlet'] == 'irish_examiner' and row['sub_excl'] == '1':
        return 'sub_exclusive'
    if row['outlet'] == 'gript' and row['gript_premium'] == '1':
        return 'gript_premium'
    if row['outlet'] == 'gript' and row['is_otd'] == '1':
        return 'gript_otd'
    body_hash = row['body_sha1']
    if body_hash:
        if body_hash in seen:
            return 'dup_body'
        seen.add(body_hash)
    norm_hash = _norm_hash(body)
    if norm_hash in seen_norm:
        return 'dup_body_norm'
    seen_norm.add(norm_hash)
    if row['period_fine'] in DROP_OOR:
        return 'out_of_range'
    return ''


def gpt_period(fine_period: str) -> str:
    """Collapse a fine period_of() label to binary pre / post (or '').

    The labels were produced in the parser but I decided not to use them.

    Args:
        fine_period (str): Specific period gauge by date from GPT release.

    Returns:
        str: Simple gauge.

    """
    if fine_period in PRE_FINE:
        return 'pre'
    if fine_period in POST_FINE:
        return 'post'
    return ''


def label_drops(rows: list[dict]) -> None:
    """Annotate each row in place with drop_reason (dedup earliest-first).

    Args:
        rows (list[dict]): Rows to have labels examined and processed for dropping.

    """
    seen: set[str] = set()
    seen_norm: set[str] = set()
    ordered_rows = sorted(
        rows, key=lambda x: (x['published_date'] or '9999', int(x['article_id']))
    )
    for row in ordered_rows:
        row['drop_reason'] = drop_reason(row, seen, seen_norm)


# Suggested by AI.
def _write_index(rows: list[dict]) -> None:
    """Write the article drop index.

    Args:
        rows (list[dict]): Rows to be examined for the index.

    """
    cols = ['article_id', 'outlet', 'period_fine', 'period', 'is_wire', 'drop_reason']
    with (DATA / 'parsed_index.csv').open('w', newline='', encoding='utf-8') as indexed:
        write = csv.writer(indexed)
        write.writerow(cols)
        write.writerows([row[col] for col in cols] for row in rows)


def main() -> int:
    """Run the curation and write the index plus the pre/post corpus."""
    with PARSED.open(encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row['period_fine'] = row['period']
        row['period'] = gpt_period(row['period'])
        row['year'] = (row['published_date'] or '')[:4]
        row['month'] = (row['published_date'] or '')[:7]
    label_drops(rows)
    _write_index(rows)

    logger.info('drop reasons: %s', dict(Counter(row['drop_reason'] for row in rows)))

    usable_rows = [row for row in rows if not row['drop_reason']]

    logger.info('usable rows: %d', len(usable_rows))

    cell: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in usable_rows:
        cell[row['outlet']][row['period']] += 1
    print(f'{"outlet":16}{"pre":>8}{"post":>8}')
    for outlet in OUTLETS:
        print(f'{outlet:16}{cell[outlet]["pre"]:>8}{cell[outlet]["post"]:>8}')
    return 0
