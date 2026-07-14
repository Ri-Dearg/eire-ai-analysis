"""Parse the articles offline to extract the article body.

Provides a fix for boilerplate-tail stripping.

Two body strings are produced per article and kept side by side on purpose:

body_raw, the exact dry-run extraction and body_text, the same content with
template  stripped. This is the detector-facing text.

The DB is read strictly read-only; raw_page is never altered. Output is a
single CSV with one row per article and no drop decisions,so the body pass runs
once and the easier curation can be re-run freely.
"""

import os
import sqlite3
import time

# Multiprocessing structure suggested and designed by AI for faster parsing.
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = Path(os.environ.get('PARSE_DB', ROOT / 'data' / 'dataset.db'))
OUT_DIR = Path(os.environ.get('PARSE_OUT', ROOT / 'data'))

N_WORKERS = 4


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
    return len(ids)


def main() -> int:
    """Run the parse pass across all workers and merge to ``OUT_CSV``."""
    current_time = time.time()
    with Pool(N_WORKERS) as pool:
        remaining = pool.map(worker, range(N_WORKERS))
