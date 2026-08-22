"""Export a headlines-only prompt set for the §1c Irish AI generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
