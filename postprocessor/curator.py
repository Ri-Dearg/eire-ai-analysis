from __future__ import annotations

import csv
from pathlib import Path

# ---------- DIRECTORIES ----------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
PARSED = DATA / 'parsed_all.csv'

PRE_FINE = ('pre',)
POST_FINE = ('straddle', 'mid', 'post')


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
