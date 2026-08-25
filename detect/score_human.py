"""Score the human calibration anchor (and RADAR's known-AI remainder) separately."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from calibrate.calibrate import (
    DETECTION_DIR as DET_DIR,
)
from calibrate.calibrate import (
    HUMAN_PARSED,
    KNOWN_AI,
)
from calibrate.calibrate import (
    human_anchor_df as human_anchor_frame,
)

HEAVY = ('fastdetectgpt', 'binoculars')


def human_frame() -> pd.DataFrame:
    """Return the usable human-anchor rows as an (id, text) frame.

    Returns:
        pd.DataFrame: Columns ``id`` and ``text``.

    """
    ok = human_anchor_frame()
    return pd.DataFrame(
        {
            'id': 'human:' + ok['outlet'] + ':' + ok['article_id'].astype(str),
            'text': ok['body_text'],
        }
    )


def ai_frame() -> pd.DataFrame:
    """Return the known-AI rows as an (id, text) frame (``ai:<model>:<n>`` ids)."""
    ai = pd.read_csv(KNOWN_AI, dtype=str).fillna('')
    return pd.DataFrame({'id': 'ai:' + ai['id'].astype(str), 'text': ai['text']})


def _pending(frame: pd.DataFrame, detector: str) -> int:
    """Return how many of ``frame``'s ids are not yet in a detector checkpoint."""
    path = DET_DIR / f'{detector}.csv'
    if not path.exists():
        return len(frame)
    done = set(pd.read_csv(path, usecols=[0]).iloc[:, 0])
    return int((~frame['id'].isin(done)).sum())


def main(argv: list[str]) -> int:
    """Score the human anchor (or ``--ai`` remainder) with the named detectors.

    Args:
        argv (list[str]): Detector names plus optional ``--ai`` / ``--dry-run``.

    Returns:
        int: 0 on success, 1 on missing inputs or bad detector names.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'detectors', nargs='+', help='detector names (see detection.detectors.DETECTORS)'
    )
    parser.add_argument(
        '--ai',
        action='store_true',
        help='score the known-AI set instead of the human anchor',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='print pending counts and exit (no model load)',
    )
    args = parser.parse_args(argv)

    src = KNOWN_AI if args.ai else HUMAN_PARSED
    if not Path(src).exists():
        hint = '' if args.ai else ' -- run the anchor parse first (Master Plan A2)'
        print(f'ERROR: {src} not found{hint}', file=sys.stderr)
        return 1
    frame = ai_frame() if args.ai else human_frame()
    label = 'known-AI' if args.ai else 'human-anchor'
    print(f'{label} rows: {len(frame):,}')

    for name in args.detectors:
        pending = _pending(frame, name)
        print(f'[{name}] pending: {pending:,}')
        if args.ai and name in HEAVY and pending:
            print(
                f'  WARNING: {name} known-AI belongs on Colab '
                '(days of Mac compute) -- continuing anyway.',
                file=sys.stderr,
            )
    if args.dry_run:
        return 0

    from detect.detect import (  # noqa: PLC0415 - deliberate lazy import
        DETECTORS,
        build_detector,
        run_detector,
    )

    unknown = [n for n in args.detectors if n not in DETECTORS]
    if unknown:
        print(
            f'unknown detectors {unknown}; choose from {list(DETECTORS)}', file=sys.stderr
        )
        return 1
    for name in args.detectors:
        print(f'[{name}] loading weights + scoring {label} ...', flush=True)
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
