"""Take a sample of URLs from the sitemap xmls to pass onto the scraper."""

import csv
from datetime import date
from pathlib import Path

from crawler import Article, _clean_url

# ---------- CONFIG ----------
PRE_GPT_NO = 1000
POST_GPT_NO = 1000


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


print(load_inventory('data/rte_inventory.csv')[:5])
