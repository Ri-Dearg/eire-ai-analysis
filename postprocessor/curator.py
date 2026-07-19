from __future__ import annotations

import csv
import re
from pathlib import Path

# ---------- DIRECTORIES ----------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
PARSED = DATA / 'parsed_all.csv'

PRE_FINE = ('pre',)
POST_FINE = ('straddle', 'mid', 'post')

MIN_BODY_CHARS = 400
STUB_MIN_WORDS = 20
NONPROSE_MIN_WORDS = 120
NONPROSE_MAX_WPS = 90

_TERM_RE = re.compile(r'[.!?…]')


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
    return 0


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


def main() -> int:
    """Run the curation and write the index plus the pre/post corpus."""
    with PARSED.open(encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row['period_fine'] = row['period']
        row['period'] = gpt_period(row['period'])
        row['year'] = (row['published_date'] or '')[:4]
        row['month'] = (row['published_date'] or '')[:7]
    return 0
