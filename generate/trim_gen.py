"""Trim every generated article back to its last complete sentence."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN_DIR = ROOT / 'data' / 'generation'
DET_DIR = ROOT / 'data' / 'detection'
GEN_GLOB = 'generated_*.csv'
DETECTORS = ('fastdetectgpt', 'binoculars', 'perplexity', 'radar')
MIN_WORDS = 100  # mirrors the notebook / API / score_gen gate

csv.field_size_limit(1 << 24)

# One definition of "sentence-final", used by both the fast path and the trim, so they
# can never disagree: terminal punctuation plus an optional closing quote or bracket.
# A body ending ".)" or '."' is complete, not cut off.
_TERMINAL = r'[.!?]["”\')\]]?'
_CLEAN_TAIL = re.compile(_TERMINAL + r'$')
# Greedy + DOTALL: matches up to the LAST terminator sitting at a whitespace-or-end
# boundary. That boundary is what stops it firing inside "EUR 1.5 million".
_SENTENCE = re.compile(r'.*' + _TERMINAL + r'(?=\s|$)', re.S)

_PREAMBLE = re.compile(
    r'^\s*(?:sure|certainly|of course|okay)\b[^\n:]{0,60}?'
    r"\bhere(?:'?s| is)\b[^\n:]{0,60}?:[ \t]*\n+",
    re.I,
)
_BEGORRAH = re.compile(r'^\s*sure and begorrah', re.I)


def strip_preamble(text: str) -> str:
    """Remove a leading conversational wrapper line from a generated body.

    A no-op on text that has none, so the script stays safe to re-run.

    Args:
        text (str): Generated article body.

    Returns:
        str: The body with any wrapper line removed.

    """
    if _BEGORRAH.match(text):
        return text  # register, not wrapper: model output stays untouched
    return _PREAMBLE.sub('', text, count=1).lstrip()


def ends_clean(text: str) -> bool:
    """Return True if the body already ends at a sentence boundary.

    Args:
        text (str): Article body.

    Returns:
        bool: Whether the text needs no trimming.

    """
    return bool(_CLEAN_TAIL.search(text.rstrip()))


def trim_to_sentence(text: str) -> str:
    """Trim a body back to its last complete sentence.

    A no-op on text that already ends cleanly, which makes the whole script safe to
    re-run over a file it has already processed.

    Args:
        text (str): Article body, possibly cut off mid-sentence.

    Returns:
        str: The trimmed body, or ``''`` if it contains no sentence boundary at all.

    """
    stripped = text.rstrip()
    if ends_clean(stripped):
        return stripped
    match = _SENTENCE.match(stripped)
    return match.group(0).rstrip() if match else ''


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
        body = trim_to_sentence(strip_preamble(original))
        if body == original.rstrip():
            kept.append(row)
            continue
        changed.add(row['id'])
        if not body:
            stats['no_sentence'] += 1
            continue
        words = len(body.split())
        if words < MIN_WORDS:
            stats['too_short'] += 1
            continue
        lost.append(len(original.split()) - words)
        row['text'] = body
        row['word_count'] = str(words)
        stats['trimmed'] += 1
        kept.append(row)

    stats['dropped'] = stats['no_sentence'] + stats['too_short']
    stats['words_lost_mean'] = round(statistics.mean(lost)) if lost else 0
    stats['words_lost_median'] = round(statistics.median(lost)) if lost else 0

    if not dry_run and changed:
        with path.open('w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(kept)
    return changed, stats


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
