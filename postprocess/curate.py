"""Curate the refined balanced analysis corpus from ``data/parsed_all.csv``.

Curation Pipeline (In order of match.):

1. Drop non-200 / no-raw.
2. Drop thin bodies (body_len_raw < 400) and empty clean bodies.
3. Drop stubs e.g. 'Listen here:'
   audio stubs. Valid short briefs (>= 20 words) are kept.
4. Drop non-prose, bodies with no sentence terminator, or long bodies whose
   words-per-sentence is implausibly high (election-results tables, photo grids).
5. Drop Examiner subscriber-exclusive rows.
6. Drop Gript premium and ON THIS DAY.
7. Dedup on the raw body sha1, then on a normalised body hash to catch templated text;
   both keep the earliest by, which also clears cross-period duplicates.
8. Drop only genuinely out-of-range rows (pre-2019 strays, future/undated).

Outputs (in data/):
* parsed_index.csv: every article id with period_fine, binary
  period, is_wire and drop_reason (empty = kept) - Advised by AI.
* corpus.csv: the refined balanced pre/post analysis corpus.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .parse import TODAY

logger = logging.getLogger(__name__)

# ---------- DIRECTORIES ----------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
PARSED = DATA / 'parsed_all.csv'

# ---------- SETTINGS ----------
SEED = 37
OUTLETS = ('gript', 'irish_examiner', 'rte', 'the_liberal')
csv.field_size_limit(1 << 24)

# ---------- TIME INTERVALS ----------
PERIODS = ('pre', 'post')
PRE_FINE = ('pre',)
POST_FINE = ('straddle', 'mid', 'post')
DROP_OOR = ('out_lo', 'out_hi', '')

# ---------- CORPUS SETTINGS ----------
CORPUS_COLS = [
    'article_id',
    'url_canonical',
    'outlet',
    'published_date',
    'year',
    'month',
    'period',
    'section',
    'author',
    'is_wire',
    'word_count',
    'body_text',
]

MIN_BODY_CHARS = 400
NONPROSE_MIN_WORDS = 120
NONPROSE_MAX_WPS = 90
STUB_MIN_WORDS = 20

# ---------- REGEX VARIABLES ----------
_TERM_RE = re.compile(r'[.!?…]')
_WS_RE = re.compile(r'\s+')


# ---------- ARTICLE DEFINING ----------
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


# ---------- ARTICLE SELECTION ----------
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


# ---------- FILE OUTPUT ----------
def build(rows: list[dict]) -> tuple[list[dict], int]:
    """Balance the usable rows into equal pre/post cells per outlet.

    The common cell size is the smallest outlet x period cell. Returns the selected
    rows and that per-cell size.

    Args:
        rows (list[dict]): rows to stratify.

    Returns:
        tuple[list[dict], int]: stratified rows, min number of cells to produce.

    """
    rng = random.Random(SEED)
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if not row['drop_reason'] and row['period'] in PERIODS:
            cells[row['outlet'], row['period']].append(row)
    target = min(len(cells[outlet, period]) for outlet in OUTLETS for period in PERIODS)
    output: list[dict] = []
    for outlet in OUTLETS:
        for period in PERIODS:
            output.extend(stratified_pick(cells[outlet, period], target, rng))
    return output, target


def stratified_pick(rows: list[dict], target: int, rng: random.Random) -> list[dict]:
    """Pick target rows from rows, stratified by year+month.

    Allocates the target across year, month in proportion to their
    availability.

    Args:
        rows (list[dict]): rows to stratify.
        target (int): Target number of rows.
        rng (random.Random): Randomised seed.

    Returns:
        list[dict]: Chosen rows to utilise.

    """
    if len(rows) <= target:
        return list(rows)
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        strata[row['year'], row['month']].append(row)
    total = len(rows)
    allocation = {key: int(target * len(value) / total) for key, value in strata.items()}
    short = target - sum(allocation.values())
    spare = sorted(strata, key=lambda k: len(strata[k]) - allocation[k], reverse=True)
    for key in spare[:short]:
        allocation[key] += 1
    picked: list[dict] = []
    for key, value in strata.items():
        picked.extend(rng.sample(value, min(allocation[key], len(value))))
    if len(picked) < target:
        pool = [row for row in rows if row not in picked]
        picked.extend(rng.sample(pool, min(target - len(picked), len(pool))))
    return picked


def write_corpus(path: Path, rows: list[dict]) -> None:
    """Write selected rows to path.

    Args:
        path (Path): Path to output file.
        rows (list[dict]): Rows to write to file.

    """
    rows = sorted(
        rows,
        key=lambda row: (
            row['outlet'],
            row['period'],
            row['published_date'],
            int(row['article_id']),
        ),
    )
    with path.open('w', newline='', encoding='utf-8') as fh:
        write = csv.DictWriter(fh, fieldnames=CORPUS_COLS, extrasaction='ignore')
        write.writeheader()
        write.writerows(rows)


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
    n_hi = sum(r['period_fine'] == 'out_hi' for r in rows)
    if n_hi:
        logger.warning(
            '%d rows dated after the TODAY snapshot (%s) were dropped '
            '-- advance TODAY or accept the boundary',
            n_hi,
            TODAY,
        )
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

    picked, target = build(rows)
    output = DATA / 'corpus.csv'
    write_corpus(output, picked)

    wire = sum(row['is_wire'] == '1' for row in picked)
    yr = Counter(row['year'] for row in picked if row['period'] == 'post')

    logger.info(
        'corpus: %dx%d cells @ %d/cell = %d rows (%d wire-flagged) -> %s',
        len(PERIODS),
        len(OUTLETS),
        target,
        len(picked),
        wire,
        output,
    )
    logger.info('  post-cell year mix: %s', dict(sorted(yr.items())))
    return 0
