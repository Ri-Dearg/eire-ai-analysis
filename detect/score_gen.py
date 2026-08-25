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
