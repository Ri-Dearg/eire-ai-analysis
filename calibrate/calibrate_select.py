"""Select the per-outlet, held-out human written articles."""

import csv
import logging
import random
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

from crawl import _clean_url
from sample.sample import OUTLETS as SAMPLER_OUTLETS

# ---------- CONFIG ----------
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
CALIBRATE_DIR = DATA / 'calibration'
CALIBRATION_DIR = DATA / 'calibration'

SEED = 37
HUMAN_TARGET = 2000  # Per outlet
START = date(2019, 1, 1)  # inclusive
RELEASE = date(2022, 11, 30)  # exclusive
OUTLETS = ('rte', 'irish_examiner', 'the_liberal', 'gript')

csv.field_size_limit(1 << 24)


# ---------- READ FILES ----------
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


def load_candidates(outlet: str) -> list[dict]:
    """Return the eligible held-out pre-ChatGPT inventory rows for one outlet.

    Args:
        outlet (str): Outlet name.

    Returns:
        list[dict]: Rows with ``url``, ``published_date``, ``year``, ``month``.

    """
    config = SAMPLER_OUTLETS[outlet]
    used_articles = excluded_clean_urls(outlet)
    output: list[dict] = []
    with (DATA / f'{outlet}_inventory.csv').open(encoding='utf-8') as file:
        for row in csv.DictReader(file):
            pub_date = row['published_date'] or ''
            if not pub_date:
                continue
            if not START <= date.fromisoformat(pub_date) < RELEASE:
                continue
            if config.category(row['url']) in config.exclude:
                continue
            if _clean_url(row['url']) in used_articles:
                continue
            output.append(
                {
                    'url': row['url'],
                    'published_date': pub_date,
                    'year': pub_date[:4],
                    'month': pub_date[:7],
                }
            )
    return output


# ---------- SELECTION ----------
def stratified_draw(rows: list[dict], target: int, rng: random.Random) -> list[dict]:
    """Draw target rows stratified by year+month.

    Args:
        rows (list[dict]): Candidate rows.
        target (int): Number to draw.
        rng (random.Random): Seeded RNG for reproducibility.

    Returns:
        list[dict]: The drawn rows.

    """
    if len(rows) <= target:
        return list(rows)
    strata: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        strata[row['month']].append(row)
    total = len(rows)
    allocation = {key: int(target * len(value) / total) for key, value in strata.items()}
    short = target - sum(allocation.values())
    spare = sorted(
        strata, key=lambda key: len(strata[key]) - allocation[key], reverse=True
    )
    for key in spare[:short]:
        allocation[key] += 1
    picked: list[dict] = []
    for key, value in strata.items():
        picked.extend(rng.sample(value, min(allocation[key], len(value))))
    return picked


# ---------- OUTPUT ----------
def write_outputs(outlet: str, drawn: list[dict]) -> None:
    """Write the draw list and the persistent exclusion log.

    Args:
        outlet (str): Outlet slug.
        drawn (list[dict]): The drawn calibration rows.

    """
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    ordered = sorted(drawn, key=lambda row: (row['published_date'], row['url']))
    payload = '\n'.join(row['url'] for row in ordered) + '\n'
    for name in (f'{outlet}_human.txt', f'{outlet}_calibration.log'):
        (CALIBRATION_DIR / name).write_text(payload, encoding='utf-8')


def main() -> int:
    """Select the human anchor for every outlet and print a count summary.

    Returns:
        int: Success of failure.

    """
    print(f'{"outlet":16}{"un-sampled pool":>16}{"drawn":>8}{"months":>8}')
    total = 0
    for outlet in OUTLETS:
        rng = random.Random(SEED)  # per-outlet seed for a stable draw
        candidate_articles = load_candidates(outlet)
        drawn_articles = stratified_draw(candidate_articles, HUMAN_TARGET, rng)
        write_outputs(outlet, drawn_articles)
        total += len(drawn_articles)
        months = len({row['month'] for row in drawn_articles})
        print(
            f'{outlet:16}{len(candidate_articles):>16,}{len(drawn_articles):>8,}{months:>8}'
        )
    print(f'{"TOTAL":16}{"":>16}{total:>8,}')
    logger.info('wrote draw lists + *_calibration.log to %s', CALIBRATION_DIR)
    return 0


if __name__ == '__main__':
    sys.exit(main())
