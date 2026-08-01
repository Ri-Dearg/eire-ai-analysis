"""Select the per-outlet, held-out human written articles."""

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
CALIB_DIR = DATA / 'calibration'

SEED = 37
HUMAN_TARGET = 2000  # Per outlet
START = date(2019, 1, 1)  # inclusive
RELEASE = date(2022, 11, 30)  # exclusive
OUTLETS = ('rte', 'irish_examiner', 'the_liberal', 'gript')
