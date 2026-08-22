"""Trim every generated article back to its last complete sentence."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN_DIR = ROOT / 'data' / 'generation'
DET_DIR = ROOT / 'data' / 'detection'
GEN_GLOB = 'generated_*.csv'
DETECTORS = ('fastdetectgpt', 'binoculars', 'perplexity', 'radar')
MIN_WORDS = 100

csv.field_size_limit(1 << 24)


def _trim_file(path: Path, *, dry_run: bool) -> tuple[set[str], dict[str, int]]:
    """Trim one generated CSV in place and report which ids changed.

    Args:
        path (Path): A ``generated_*.csv``.
        dry_run (bool): If True, compute and report but write nothing.

    Returns:
        tuple[set[str], dict[str, int]]: Ids whose text changed or were dropped, and a
        counts dict for reporting.

    """
    with path.open(encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        rows = list(reader)

    changed: set[str] = set()
    kept: list[dict[str, str]] = []
    lost: list[int] = []
    stats = {'rows': len(rows), 'trimmed': 0, 'no_sentence': 0, 'too_short': 0}

    for row in rows:
        original = row['text']
        body = original


def main(argv: list[str]) -> int:
    """Trim the generated set and purge the scores of every row that changed.

    Args:
        argv (list[str]): CLI arguments.

    Returns:
        int: 0 on success, 1 if there is nothing to work on.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dry-run', action='store_true', help='report what would change, write nothing'
    )
    args = parser.parse_args(argv)

    paths = sorted(GEN_DIR.glob(GEN_GLOB))
    if not paths:
        print(f'ERROR: no {GEN_GLOB} in {GEN_DIR}', file=sys.stderr)
        return 1

    backup = ROOT / 'data' / 'backups' / 'pre_trim'
    if not args.dry_run:
        backup.mkdir(parents=True, exist_ok=True)
        for path in [*paths, *(DET_DIR / f'{d}.csv' for d in DETECTORS)]:
            if path.exists():
                shutil.copy2(path, backup / path.name)  # copy, never move
        print(f'backed up {len(list(backup.iterdir()))} files -> {backup}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
