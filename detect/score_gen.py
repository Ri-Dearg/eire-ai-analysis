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
