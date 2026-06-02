"""Take a sample of URLs from the sitemap xmls to pass onto the scraper."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from crawler import Article, _clean_url

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# ---------- CONFIG ----------
DATA_DIR = Path('./data')

PRE_GPT_NO = 1000
POST_GPT_NO = 1000

# ---------- OUTLET SETTINGS ----------
# ----- RTE: category is the segment after /news/--
_RTE_CATEGORY_RE = re.compile(r'/news/([^/]+)/')
RTE_EXCLUDE = frozenset({'business', 'weather-summary'})  # wire-fed / templated


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


# ---------- SAMPLE ----------
def sample(outlet_slug: str, data_dir: Path) -> None:
    inventory = load_inventory(data_dir / f'{outlet_slug}_inventory.csv')

    print(inventory[:5])


sample(OUTLETS['rte'].slug, DATA_DIR)
