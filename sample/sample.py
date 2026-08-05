"""Take a sample of URLs from the sitemap xmls to pass onto the scraper."""

from __future__ import annotations

import csv
import logging
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from crawl import Article, _clean_url

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# ---------- CONFIG ----------
DATA_DIR = Path('./data')

# No of articles to sample from each period.
PRE_GPT_NO = 0
POST_GPT_NO = 2500


if PRE_GPT_NO and not os.environ.get('SAMPLE_ALLOW_PRE'):
    MESSAGE = (
        'PRE_GPT_NO > 0 will change the calibration anchor draw. '
        'Set SAMPLE_ALLOW_PRE=1 if that is intended.'
    )
    raise SystemExit(MESSAGE)

# Select years for sampling.
# (Gript starts 2019).
PRE_YEARS = range(2019, 2023)  # 2019-2022
POST_YEARS = range(2023, 2024)

# RNG seed for a reproducible sample across runs.
SEED = 37

# ---------- OUTLET SETTINGS ----------
# ----- RTE: category is the segment after /news/--
_RTE_CATEGORY_RE = re.compile(r'/news/([^/]+)/')
# Sport, business, and weather-summary are formulaic match/market register.
RTE_EXCLUDE = frozenset({'business', 'weather-summary'})

# ----- Irish Examiner: category is the first path segment -----
_EXAMINER_CATEGORY_RE = re.compile(r'https?://[^/]+/([^/]+)/')
# Sport and business (incl. columnist variants) are formulaic match/market
# register; property mixes editorial features with templated estate-agent
# listings that can't be split by URL; the rest are sponsored/staging.
EXAMINER_EXCLUDE = frozenset(
    {
        'sport',
        'sport-columnists',
        'sport-columnists-gaa',
        'sport-columnists-golf',
        'sport-columnists-racing',
        'sport-columnists-rugby',
        'sport-columnists-soccer',
        'business',
        'business-columnists',
        'property',
        'sponsored',
        'sponsored-showcase',
        'special-reports',
        'competition',
        'morningbriefing',
        'nfytest',
        'iectrial2',
        'iectrial3',
        'xml',
        'pages',
        'podcasts-app',
        'puzzles',
        'others',
    }
)


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
    # Regex match for rte url categories
    match = _RTE_CATEGORY_RE.search(url)
    if not match:
        return 'no-category'
    category = match.group(1)
    return 'no-category' if category.isdigit() else category


def _examiner_category(url: str) -> str:
    """Read the Irish Examiner section from a URL.

    Args:
        url (str): Article URL.

    Returns:
        str: The first path segment, or 'no-category' if none.

    """
    match = _EXAMINER_CATEGORY_RE.match(url)
    return match.group(1) if match else 'no-category'


def _keep_all(_url: str) -> str:
    """Category reader for flat-slug outlets (Gript, The Liberal).

    They have no category segment and an empty exclude set, so every URL maps to
    one constant bucket that is never excluded.

    Args:
        _url (str): Article URL (unused).

    Returns:
        str: The constant 'article'.

    """
    return 'article'


# Configuration for each outlet for regex url matching.
OUTLETS: dict[str, OutletConfig] = {
    'gript': OutletConfig('gript', _keep_all),
    'irish_examiner': OutletConfig(
        'irish_examiner', _examiner_category, EXAMINER_EXCLUDE
    ),
    'rte': OutletConfig('rte', _rte_category, RTE_EXCLUDE),
    'the_liberal': OutletConfig('the_liberal', _keep_all),
}


# ---------- LOAD ----------
def _load_inventory(csv_path: str | Path) -> list[Article]:
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


def _load_sampled_log(log_path: str | Path) -> set[str]:
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
def _filter_candidates(
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
    # Filter out articles in excluded categories, and any URL already sampled.
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
    # Loop through articles, putting them into year-months.
    for article in articles:
        if article.pub_date.year in years:
            by_month[article.pub_date.isoformat()[:7]].append(article)
    logger.info(
        '%d monthly strata after year selection',
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
    # Remaining picks to allocate, updated each loop.
    remaining = min(total_wanted, sum(available_articles.values()))
    # Loop through the months, giving each an even share.
    while remaining > 0:
        active_months = [
            month
            for month in available_articles
            if available_articles[month] - article_collection[month] > 0
        ]
        # If no active months, we can't allocate any more.
        if not active_months:
            break
        # Give each active month an even share of the remaining picks.
        article_per_month = max(1, remaining // len(active_months))
        # Loop through active months in a fixed order, giving each its share.
        for month in sorted(active_months):
            if remaining == 0:
                break
            give = min(
                article_per_month,
                available_articles[month] - article_collection[month],
                remaining,
            )
            # Update the collection and remaining picks.
            article_collection[month] += give
            remaining -= give
    return article_collection


def _sample_stratified(
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
    # Group articles into year-month buckets, and count availability.
    monthly_samples = _stratify_by_month(articles, years)
    available_articles = sum(len(bucket) for bucket in monthly_samples.values())
    # Warn if we can't fill the requested total.
    if available_articles < total_wanted:
        logger.warning(
            '%s: only %d articles available, wanted %d',
            label,
            available_articles,
            total_wanted,
        )
    # Allocate the total across months as evenly as possible.
    monthly_counts = {month: len(articles) for month, articles in monthly_samples.items()}
    monthly_distribution = _spread(total_wanted, monthly_counts)
    selected_articles: list[Article] = []
    # Loop through the months in a fixed order, sampling from each as allocated.
    for month in sorted(monthly_samples):
        bucket = sorted(monthly_samples[month], key=lambda article: article.url)
        selected_articles.extend(seeded_rng.sample(bucket, monthly_distribution[month]))
    return selected_articles


# ---------- OUTPUT ----------
def _write_sample(
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
    # Sort the sample by publication date (newest first).
    ordered = sorted(
        final_sample,
        key=lambda article: (article.pub_date, article.url),
        reverse=True,
    )
    # Write the sample URLs to a .txt file, one per line.
    txt_path = output_dir / f'{slug}_sample.txt'
    txt_path.write_text(
        '\n'.join(article.url for article in ordered) + '\n', encoding='utf-8'
    )

    # Write the sample metadata to a .csv file, one article per row.
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


def _append_sampled_log(
    final_sample: Sequence[Article],
    slug: str,
    output_dir: str | Path,
) -> Path:
    """Append this run's URLs to the persistent sampled-log.

    Args:
        final_sample (Sequence[Article]): Articles drawn this run.
        slug (str): Outlet slug used in the log filename.
        output_dir (str | Path): Directory holding the log.

    Returns:
        Path: Path to the sampled-log.

    """
    log_path = Path(output_dir) / f'{slug}_sampled.log'
    with log_path.open('a', encoding='utf-8') as file:
        for article in final_sample:
            file.write(article.url + '\n')
    return log_path


# ---------- SUMMARY ----------
def _summarise(slug: str, final_sample: Sequence[Article]) -> None:
    """Log the sample's pre/post totals and per-year counts.

    Args:
        slug (str): Outlet slug, for the log line.
        final_sample (Sequence[Article]): The drawn sample.

    """
    # Log the pre/post totals, and count per year.
    pre = sum(article.period == 'pre' for article in final_sample)
    post = sum(article.period == 'post' for article in final_sample)
    logger.info('%s: %d total (%d pre, %d post)', slug, len(final_sample), pre, post)
    by_year: dict[int, int] = {}
    # Loop through the sample, counting articles per publication year.
    for article in final_sample:
        by_year[article.pub_date.year] = by_year.get(article.pub_date.year, 0) + 1
    for year in sorted(by_year):
        logger.info('  %d: %d', year, by_year[year])


def sample(outlet: OutletConfig, data_dir: Path) -> None:
    """Load, filter, stratify-sample, and write outputs for one outlet.

    Args:
        outlet (Outlet): Outlet to sample.
        data_dir (Path): Directory holding the inventory, log, and outputs.

    """
    slug = outlet.slug
    # Load the inventory and already-sampled URLs.
    inventory = _load_inventory(data_dir / f'{slug}_inventory.csv')
    already_sampled = _load_sampled_log(data_dir / f'{slug}_sampled.log')
    # The detection stage's calibration URLs must be kept separate from the
    # corpus.
    already_sampled |= _load_sampled_log(
        data_dir / 'calibration' / f'{slug}_calibration.log'
    )
    logger.info(
        '%s: loaded %d inventory articles, %d already sampled',
        slug,
        len(inventory),
        len(already_sampled),
    )

    # Filter the inventory to get options.
    filtered = _filter_candidates(inventory, outlet, already_sampled)
    logger.info(
        '%s: %d articles after category filter and sampled exclusion',
        slug,
        len(filtered),
    )

    # Use a seeded RNG for a reproducible sample across runs.
    seeded_rng = random.Random(SEED)
    # Sample from the filtered options.
    pre_sample = _sample_stratified(
        filtered,
        PRE_YEARS,
        PRE_GPT_NO,
        seeded_rng,
        'pre',
    )
    post_sample = _sample_stratified(
        filtered,
        POST_YEARS,
        POST_GPT_NO,
        seeded_rng,
        'post',
    )

    final_sample = pre_sample + post_sample

    # Write the sample outputs and update the log.
    txt_file, csv_file = _write_sample(final_sample, slug, data_dir)
    log_file = _append_sampled_log(final_sample, slug, data_dir)
    _summarise(slug, final_sample)

    logger.info(
        '%s: wrote %s (%d urls), %s, and updated %s',
        slug,
        txt_file,
        len(final_sample),
        csv_file,
        log_file,
    )


def main() -> None:
    """Run the sampling process for each outlet."""
    for outlet in OUTLETS.values():
        sample(outlet, DATA_DIR)
