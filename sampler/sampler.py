"""Take a sample of URLs from the sitemap xmls to pass onto the scraper."""

from __future__ import annotations

import csv
import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from crawler import Article, _clean_url

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# ---------- CONFIG ----------
DATA_DIR = Path('./data')

PRE_GPT_NO = 100
POST_GPT_NO = 100

# Select years for sampling.
# (Gript starts 2019).
PRE_YEARS = range(2019, 2023)  # 2019-2022
POST_YEARS = range(2025, 2027)

SEED = 37

# ---------- OUTLET SETTINGS ----------
# ----- RTE: category is the segment after /news/--
_RTE_CATEGORY_RE = re.compile(r'/news/([^/]+)/')
RTE_EXCLUDE = frozenset({'business', 'weather-summary'})


@dataclass(frozen=True)
class OutletConfig:
    """Category reading and exclusions settings.

    Attributes:
        slug (str): Names the inventory, sample, and log files.
        category (Callable[[str], str]): Maps an article URL to its category.
        exclude (frozenset[str]): Categories to drop (empty = keep everything).

    """

    slug: str
    category: Callable[[str], str]
    exclude: frozenset[str] = frozenset()


def _rte_category(url: str) -> str:
    """Read the RTE news category from a URL.

    Args:
        url (str): Article URL.

    Returns:
        str: Segment after ``/news/``, or 'no-category' for the dateless-segment
            (real news) URL form.

    """
    match = _RTE_CATEGORY_RE.search(url)
    if not match:
        return 'no-category'
    category = match.group(1)
    return 'no-category' if category.isdigit() else category


OUTLETS: dict[str, OutletConfig] = {
    'rte': OutletConfig('rte', _rte_category, RTE_EXCLUDE),
}


# ---------- LOAD ----------
def load_inventory(csv_path: str | Path) -> list[Article]:
    """Read a crawler inventory CSV into a list of articles.

    Args:
        csv_path (str | Path): Path to a ``<outlet>_inventory.csv``.

    Returns:
        list[Article]: One article per inventory row.

    """
    with Path(csv_path).open(encoding='utf-8') as file:
        return [
            Article(
                url=row['url'],
                clean_url=_clean_url(row['url']),
                pub_date=date.fromisoformat(row['published_date']),
                period=row['period'],
            )
            for row in csv.DictReader(file)
        ]


def load_sampled_log(log_path: str | Path) -> set[str]:
    """Read the set of URLs already sampled in previous runs.

    Args:
        log_path (str | Path): Path to a ``<slug>_sampled.log``.

    Returns:
        set[str]: URLs to exclude this run; empty if the log is absent.

    """
    path = Path(log_path)
    if not path.exists():
        return set()
    return {line for line in path.read_text(encoding='utf-8').splitlines() if line}


# ---------- FILTER ----------
def filter_candidates(
    articles: Sequence[Article],
    config: OutletConfig,
    already_sampled: set[str],
) -> list[Article]:
    """Drop excluded categories and any URL already sampled.

    Args:
        articles (Sequence[Article]): All inventory articles.
        config (OutletConfig): Outlet's category reader and exclude set.
        already_sampled (set[str]): URLs drawn in previous runs.

    Returns:
        list[Article]: Eligible candidates for this run.

    """
    return [
        article
        for article in articles
        if config.category(article.url) not in config.exclude
        and article.url not in already_sampled
    ]


def _stratify_by_month(
    articles: Sequence[Article],
    years: range,
) -> dict[str, list[Article]]:
    """Group articles within a year window into year-month strata.

    Args:
        articles (Sequence[Article]): Candidate articles for one period.
        years (range): Inclusive year window to keep.

    Returns:
        dict[str, list[Article]]: Articles keyed by 'YYYY-MM'.

    """
    by_month: dict[str, list[Article]] = defaultdict(list)
    total = 0
    for article in articles:
        if article.pub_date.year in years:
            total += 1
            by_month[article.pub_date.isoformat()[:7]].append(article)
    logger.info(
        '%s: %d monthly strata after year selection',
        OUTLETS['rte'].slug,
        len(by_month),
    )
    return by_month


# ---------- SAMPLE ----------
def _spread(total_wanted: int, available_articles: dict[str, int]) -> dict[str, int]:
    """Spread a total across strata as evenly as availability allows.

    Args:
        total_wanted (int): Number of picks to allocate.
        available_articles (dict[str, int]): Available count per stratum key.

    Returns:
        dict[str, int]: Picks allocated per stratum key.

    """
    article_collection = dict.fromkeys(available_articles, 0)
    remaining = min(total_wanted, sum(available_articles.values()))
    while remaining > 0:
        active_months = [
            month
            for month in available_articles
            if available_articles[month] - article_collection[month] > 0
        ]
        if not active_months:
            break
        article_per_month = max(1, remaining // len(active_months))
        for month in sorted(active_months):
            if remaining == 0:
                break
            give = min(
                article_per_month,
                available_articles[month] - article_collection[month],
                remaining,
            )
            article_collection[month] += give
            remaining -= give
    return article_collection


def sample_stratified(
    articles: Sequence[Article],
    years: range,
    total_wanted: int,
    seeded_rng: random.Random,
    label: str,
) -> list[Article]:
    """Sample articles spread evenly across year-month strata.

    Args:
        articles (Sequence[Article]): Candidates for one period.
        years (range): Inclusive year window to sample from.
        total_wanted (int): Number of articles to draw.
        seeded_rng (random.Random): Seeded RNG, for a reproducible draw.
        label (str): Period label, for the short-pool warning.

    Returns:
        list[Article]: The drawn articles for this period.

    """
    monthly_samples = _stratify_by_month(articles, years)
    available_articles = sum(len(bucket) for bucket in monthly_samples.values())
    if available_articles < total_wanted:
        logger.warning(
            '%s: only %d articles available, wanted %d',
            label,
            available_articles,
            total_wanted,
        )
    monthly_counts = {month: len(articles) for month, articles in monthly_samples.items()}
    monthly_distribution = _spread(total_wanted, monthly_counts)
    selected_articles: list[Article] = []
    for month in sorted(monthly_samples):
        bucket = sorted(monthly_samples[month], key=lambda article: article.url)
        selected_articles.extend(seeded_rng.sample(bucket, monthly_distribution[month]))
    return selected_articles


# ---------- OUTPUT ----------
def write_sample(
    final_sample: Sequence[Article],
    slug: str,
    out_dir: str | Path,
) -> tuple[Path, Path]:
    """Write the sample URL list (.txt), (.csv).

    Args:
        final_sample (Sequence[Article]): Sampled articles.
        slug (str): Outlet slug used in the output filenames.
        out_dir (str | Path): Directory to write into; created if needed.

    Returns:
        tuple[Path, Path]: Paths to the written (txt, csv) files.

    """
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        final_sample,
        key=lambda article: (article.pub_date, article.url),
        reverse=True,
    )

    txt_path = output_dir / f'{slug}_sample.txt'
    txt_path.write_text(
        '\n'.join(article.url for article in ordered) + '\n', encoding='utf-8'
    )

    csv_path = output_dir / f'{slug}_sample.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(('url', 'published_date', 'year', 'period'))
        for article in ordered:
            writer.writerow(
                (
                    article.url,
                    article.pub_date.isoformat(),
                    article.pub_date.year,
                    article.period,
                )
            )

    return txt_path, csv_path


def sample(outlet: OutletConfig, data_dir: Path) -> None:
    slug = outlet.slug
    inventory = load_inventory(data_dir / f'{slug}_inventory.csv')
    already_sampled = load_sampled_log(data_dir / f'{slug}_sampled.log')
    filtered = filter_candidates(inventory, outlet, already_sampled)
    logger.info(
        '%s: loaded %d inventory articles, %d already sampled',
        slug,
        len(inventory),
        len(already_sampled),
    )

    seeded_rng = random.Random(SEED)
    pre_sample = sample_stratified(
        filtered,
        PRE_YEARS,
        PRE_GPT_NO,
        seeded_rng,
        'pre',
    )
    post_sample = sample_stratified(
        filtered,
        POST_YEARS,
        POST_GPT_NO,
        seeded_rng,
        'post',
    )

    final_sample = pre_sample + post_sample

    write_sample(final_sample, slug, data_dir)


sample(OUTLETS['rte'], DATA_DIR)
