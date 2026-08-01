"""Select the per-outlet, held-out human written articles."""

import csv
import random
from datetime import date
from pathlib import Path

from crawl import _clean_url
from sample.sample import OUTLETS as SAMPLER_OUTLETS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
CALIB_DIR = DATA / 'calibration'

SEED = 37
HUMAN_TARGET = 2000  # Per outlet
START = date(2019, 1, 1)  # inclusive
RELEASE = date(2022, 11, 30)  # exclusive
OUTLETS = ('rte', 'irish_examiner', 'the_liberal', 'gript')
csv.field_size_limit(1 << 24)


def _read_lines(path: Path) -> set[str]:
    """Return the non-empty lines of ``path`` as a set (empty if absent).

    Args:
        path (Path): Path to file to read.

    Returns:
        set[str]: Non-empty lines from the file.

    """
    if not path.exists():
        return set()
    return {line for line in path.read_text(encoding='utf-8').splitlines() if line}


def excluded_clean_urls(outlet: str) -> set[str]:
    """Return the set of cleaned URLs already used by the corpus, for one outlet.

    Args:
        outlet (str): Outlet name.

    Returns:
        set[str]: Cleaned URLs to exclude from calibration.

    """
    raw = _read_lines(DATA / f'{outlet}_sampled.log')
    if outlet == 'gript':
        raw |= _read_lines(DATA / 'gript_premium.log')
    clean = {_clean_url(url) for url in raw}
    corpus = DATA / 'corpus.csv'
    if corpus.exists():
        with corpus.open(encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader)
            index_url = header.index('url_canonical')
            index_outlet = header.index('outlet')
            clean |= {
                _clean_url(row[index_url])
                for row in reader
                if row[index_outlet] == outlet
            }
    return clean


def load_candidates(outlet: str):
    config = SAMPLER_OUTLETS[outlet]
    used_articles = excluded_clean_urls(outlet)
    print(used_articles)


def main() -> int:
    """Select the human anchor for every outlet and print a count summary."""
    print(f'{"outlet":16}{"un-sampled pool":>16}{"drawn":>8}{"months":>8}')
    total = 0
    for outlet in OUTLETS:
        rng = random.Random(SEED)  # per-outlet seed for a stable draw
        load_candidates(outlet)
    return 0
