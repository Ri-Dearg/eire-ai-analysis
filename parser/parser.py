"""Parse the articles offline to extract the article body.

Provides a fix for boilerplate-tail stripping.

Two body strings are produced per article and kept side by side on purpose:

body_raw, the exact dry-run extraction and body_text, the same content with
template  stripped. This is the detector-facing text.

The DB is read strictly read-only; raw_page is never altered. Output is a
single CSV with one row per article and no drop decisions,so the body pass runs
once and the easier curation can be re-run freely.
"""

import csv
import os
import re
import sqlite3
import time

# Multiprocessing structure suggested and designed by AI for faster parsing.
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = Path(os.environ.get('PARSE_DB', ROOT / 'data' / 'dataset.db'))
OUT_DIR = Path(os.environ.get('PARSE_OUT', ROOT / 'data'))
HTTP_OK = 200


N_WORKERS = 4

BATCH = 200
TIME_BUDGET = float(os.environ.get('PARSE_BUDGET', '0'))

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


def feat_rte(html: str, url: str) -> dict:
    """Extract RTE fields; date falls back to URL path for LD-less live pages."""
    return html  # Placeholder for actual feature extraction logic


def _row_record(row: tuple) -> dict:
    """Build one record from a (id, outlet, url, curl, status, raw) row."""
    aid, outlet, aurl, curl, status, raw = row
    record = dict.fromkeys(COLS, '')
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
        features = feat_rte(raw, curl or aurl)
    except Exception as exc:
        record['parse_error'] = type(exc).__name__
        return record
    return record


def worker(wid: int) -> int:
    """Parse the id-stripe id %% N_WORKERS == wid into part_<wid>.csv."""
    connection = sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=60)
    cursor = connection.cursor()
    ids = [
        row[0]
        for row in cursor.execute(
            f'SELECT id FROM article WHERE (id % {N_WORKERS})={wid} ORDER BY id'
        )
    ]

    path = OUT_DIR / f'part_{wid}.csv'
    last_done = 0
    if path.exists():
        with path.open(encoding='utf-8') as part:
            for line in part:
                head = line.split(',', 1)[0]
                if head.isdigit():
                    last_done = max(last_done, int(head))

    new_file = last_done == 0
    ids = [id for id in ids if id > last_done]
    current_time = time.time()
    num = 0
    done = True

    with path.open('a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        if wid == 0 and new_file:
            w.writeheader()
        for i in range(0, len(ids), BATCH):
            if TIME_BUDGET and time.time() - current_time > TIME_BUDGET:
                done = False
                break
            chunk = ids[i : i + BATCH]
            placeholders = ','.join('?' * len(chunk))
            query = (
                'SELECT a.id, o.name, a.url, a.url_canonical, r.http_status, '
                'r.raw_html FROM article a JOIN outlet o ON o.id=a.outlet_id '
                'LEFT JOIN raw_page r ON r.article_id=a.id '
                f'WHERE a.id IN ({placeholders})'
            )
            for db_row in cursor.execute(query, chunk):
                w.writerow(_row_record(db_row))
                num += 1
            fh.flush()

    return len(ids)


def main() -> int:
    """Run the parse pass across all workers and merge to ``OUT_CSV``."""
    current_time = time.time()
    with Pool(N_WORKERS) as pool:
        remaining = pool.map(worker, range(N_WORKERS))
