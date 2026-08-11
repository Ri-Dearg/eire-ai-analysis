"""Fetch the human calibration articles into a separate, isolated database."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from scrape.gript_rest import ingest_gript
from scrape.scrape import ingest

# ---------- CONFIG ----------
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_DIR = ROOT / 'data' / 'calibration'
CAL_DB = str(CALIBRATION_DIR / 'calibration.db')
SOURCE_FEED = 'calibration'
LEGACY = ('rte', 'irish_examiner', 'the_liberal')


# ---------- FETCH ----------
def _urls(outlet: str) -> list[str]:
    """Return the drawn calibration URLs for one outlet (one per line).

    Args:
        outlet (str): The outlet name

    Returns:
        list[str]: List of urls.

    """
    path = CALIBRATION_DIR / f'{outlet}_human.txt'
    return [line for line in path.read_text(encoding='utf-8').splitlines() if line]


def main() -> int:
    """Ingest every outlet's calibration URLs into the isolated calibration DB.

    Returns:
        int: Success or failure.

    """
    if not Path(CAL_DB).exists():
        logger.error(
            'ERROR: %s not found. Create it first (schema + outlets) -- see RUNBOOK.md.',
            CAL_DB,
        )
        return 1
    for outlet in LEGACY:
        urls = _urls(outlet)
        logger.info('%s: ingesting %d calibration URLs -> %s', outlet, len(urls), CAL_DB)
        counts = ingest(urls, outlet, SOURCE_FEED, db_path=CAL_DB)
        logger.info('  %s: %s', outlet, counts)
    gript_urls = _urls('gript')
    logger.info(
        'gript: ingesting %d calibration URLs (REST) -> %s', len(gript_urls), CAL_DB
    )
    logger.info('  gript: %s', ingest_gript(gript_urls, db_path=CAL_DB))
    return 0


if __name__ == '__main__':
    sys.exit(main())
