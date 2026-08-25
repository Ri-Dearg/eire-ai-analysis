"""Score the generated Irish-register AI set, incrementally as Colab produces it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from calibrate.calibrate import DETECTION_DIR as DET_DIR

ROOT = Path(__file__).resolve().parent.parent
GEN_DIR = ROOT / 'data' / 'generation'
GEN_GLOB = 'generated_*.csv'
MIN_WORDS = 100  # mirrors the notebook/API usable() gate


def gen_frame() -> pd.DataFrame:
    """Return all generated Irish-register rows as an (id, text) frame.

    Returns:
        pd.DataFrame: Columns ``id`` and ``text`` (plus ``model`` for reporting).

    """
    paths = sorted(GEN_DIR.glob(GEN_GLOB))
    if not paths:
        return pd.DataFrame(columns=['id', 'text', 'model'])
    frames = []
    for path in paths:
        frame = pd.read_csv(path, dtype=str).fillna('')
        missing = {'id', 'text'} - set(frame.columns)
        if missing:
            print(f'  WARN: {path.name} lacks {missing}, skipped', file=sys.stderr)
            continue
        frame['_src'] = path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=['id', 'text', 'model'])
    all_rows = pd.concat(frames, ignore_index=True)
    usable = all_rows[all_rows['text'].str.split().str.len() >= MIN_WORDS]
    deduped = usable.drop_duplicates(subset=['id'], keep='first')
    cols = ['id', 'text'] + [c for c in ('model', '_src') if c in deduped.columns]
    return deduped[cols].reset_index(drop=True)


def _pending(frame: pd.DataFrame, detector: str) -> int:
    """Return how many of ``frame``'s ids are not yet in a detector checkpoint.

    Args:
        frame (pd.DataFrame): The generated-set frame.
        detector (str): Detector name.

    Returns:
        int: Count of unscored ids.

    """
    path = DET_DIR / f'{detector}.csv'
    if not path.exists():
        return len(frame)
    done = set(pd.read_csv(path, usecols=[0]).iloc[:, 0])
    return int((~frame['id'].isin(done)).sum())


def main(argv: list[str]) -> int:
    """Score the generated Irish-register set with the named detectors.

    Args:
        argv (list[str]): Detector names (default: all) plus optional ``--dry-run``.

    Returns:
        int: 0 on success, 1 on missing inputs or bad detector names.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'detectors',
        nargs='*',
        help='detector names (default: all four)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='print pending counts and exit (no model load)',
    )
    args = parser.parse_args(argv)

    frame = gen_frame()
    if frame.empty:
        print(
            f'ERROR: no usable rows in {GEN_DIR}/{GEN_GLOB} -- download the '
            'generated CSV from Colab (or run generate_frontier_api.py) first.',
            file=sys.stderr,
        )
        return 1
    print(f'generated rows: {len(frame):,}')
    if 'model' in frame.columns:
        for model, count in frame['model'].value_counts().sort_index().items():
            print(f'  {model:<16} {count:>5}')

    names = args.detectors or ['perplexity', 'radar', 'fastdetectgpt', 'binoculars']
    for name in names:
        print(f'[{name}] pending: {_pending(frame, name):,}')
    if args.dry_run:
        return 0

    # Torch import deferred so --dry-run works without a GPU environment.
    from detect.detect import (  # noqa: PLC0415 - deliberate lazy import
        DETECTORS,
        build_detector,
        run_detector,
    )

    unknown = [n for n in names if n not in DETECTORS]
    if unknown:
        print(
            f'unknown detectors {unknown}; choose from {list(DETECTORS)}',
            file=sys.stderr,
        )
        return 1
    for name in names:
        print(f'[{name}] loading weights + scoring generated set ...', flush=True)
        detector = build_detector(name)
        out = run_detector(
            detector,
            frame['id'].tolist(),
            frame['text'].tolist(),
            DET_DIR / f'{name}.csv',
        )
        print(f'[{name}] done -> {out}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
