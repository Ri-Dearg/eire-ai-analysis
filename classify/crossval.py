"""Cross-validated per-generator recall with confidence intervals."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from statsmodels.stats.proportion import proportion_confint

logger = logging.getLogger(__name__)

# ---------- FILE SYSTEM ----------
ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT / 'data' / 'classify'
REPORT = SPLIT_DIR / 'crossval_report.csv'

# ---------- SETTINGS ----------
SEED = 37
FOLDS = 5
AI_LABEL = 1
HUMAN_LABEL = 0
CONFIDENCE = 0.05
MODELS = os.environ.get('CROSSVAL_MODEL', 'tfidf,frozen').split(',')

# Frozen-encoder settings, held identical to classify/frozen.py so the two are comparable.
ENCODER = 'distilbert-base-uncased'
MAX_TOKENS = 256
BATCH_SIZE = 16
