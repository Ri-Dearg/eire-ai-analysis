"""Export a headlines-only prompt set for the §1c Irish AI generation."""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import sys
from pathlib import Path

from bs4 import BeautifulSoup

# ---------- CONFIG ----------
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / 'data' / 'dataset.db'
CORPUS = ROOT / 'data' / 'corpus.csv'
OUT_DIR = ROOT / 'data' / 'generation'
OUT_CSV = OUT_DIR / 'gen_prompts.csv'

SEED = 37
DEFAULT_PER_CELL = 75
MIN_HEADLINE_CHARS = 15
# Trailing " | Site", " - Site", " » Site" boilerplate to strip from <title>.
_SEPARATORS = (' | ', ' - ', ' — ', ' » ', ' :: ')


# ---------- CLEAN ----------
def _clean_headline(text: str) -> str:
    """Unescape entities, collapse whitespace, drop a trailing site suffix.

    Args:
        text (str): Raw ``og:title`` / ``<title>`` string.

    Returns:
        str: The cleaned headline (empty if nothing usable remains).

    """
    cleaned = html.unescape(text or '').strip()
    cleaned = ' '.join(cleaned.split())
    for sep in _SEPARATORS:
        if sep in cleaned:
            head = cleaned.rsplit(sep, 1)[0].strip()
            if len(head) >= MIN_HEADLINE_CHARS:
                cleaned = head
    return cleaned


def _headline(raw: str) -> str:
    """Read a headline from stored page content (JSON for Gript, else HTML).

    Args:
        raw (str): The stored ``raw_page.raw_html``.

    Returns:
        str: The headline, or '' if none could be read.

    """
    if raw and raw.lstrip()[:1] in ('{', '['):
        head = _headline_from_json(raw)
        if len(head) >= MIN_HEADLINE_CHARS:
            return head
    return _headline_from_html(raw)


def _headline_from_html(raw_html: str) -> str:
    """Extract the article headline from stored page HTML (``og:title`` first).

    Args:
        raw_html (str): The stored ``raw_page.raw_html``.

    Returns:
        str: The headline, or '' if none could be read.

    """
    soup = BeautifulSoup(raw_html, 'html.parser')
    og = soup.find('meta', attrs={'property': 'og:title'})
    if og and og.get('content'):
        head = _clean_headline(og['content'])
        if len(head) >= MIN_HEADLINE_CHARS:
            return head
    if soup.title and soup.title.string:
        return _clean_headline(soup.title.string)
    return ''


def _headline_from_json(raw: str) -> str:
    """Extract the headline from a stored WP REST JSON dump (Gript).

    Args:
        raw (str): The stored ``raw_page.raw_html`` (a JSON string).

    Returns:
        str: The cleaned headline, or '' if none could be read.

    """
    try:
        node = json.loads(raw)
    except (ValueError, TypeError):
        return ''
    title = node.get('title') if isinstance(node, dict) else None
    if isinstance(title, dict):
        title = title.get('rendered', '')
    if not isinstance(title, str):
        return ''
    return ' '.join(html.unescape(title).split())


# ---------- SAMPLE ----------
def _stratified_ids(per_cell: int) -> list[dict[str, str]]:
    """Draw a seeded, balanced sample of corpus rows across outlet x period.

    Args:
        per_cell (int): Rows to draw from each outlet x period cell.

    Returns:
        list[dict[str, str]]: Sampled rows (article_id/outlet/period/section).

    """
    cells: dict[tuple[str, str], list[dict[str, str]]] = {}
    with CORPUS.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            key = (row['outlet'], row['period'])
            cells.setdefault(key, []).append(
                {
                    'article_id': row['article_id'],
                    'outlet': row['outlet'],
                    'period': row['period'],
                    'section': row['section'],
                }
            )
    rng = random.Random(SEED)
    picked: list[dict[str, str]] = []
    for key in sorted(cells):
        pool = cells[key]
        rng.shuffle(pool)
        picked.extend(pool[:per_cell])
    return picked


def main(argv: list[str]) -> int:
    """Build ``gen_prompts.csv`` from a stratified corpus sample.

    Args:
        argv (list[str]): CLI args (``--per-cell``).

    Returns:
        int: 0 on success, 1 if inputs are missing.

    """
    # Arguments added by AI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--per-cell',
        type=int,
        default=DEFAULT_PER_CELL,
        help='articles drawn per outlet x period cell',
    )
    args = parser.parse_args(argv)
    if not DB_PATH.exists() or not CORPUS.exists():
        print(f'ERROR: need {DB_PATH} and {CORPUS}', file=sys.stderr)
        return 1

    sample = _stratified_ids(args.per_cell)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
