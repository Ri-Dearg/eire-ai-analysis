"""False-positive rate of the shipped classifiers on the pre-ChatGPT corpus cell."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from calibrate.calibrate import LENGTH_BANDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from classify.dataset import harmonise

logger = logging.getLogger(__name__)

# ---------- FILE SYSTEM ----------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
CORPUS = DATA / 'corpus.csv'
SPLIT_DIR = DATA / 'classify'
REPORT = SPLIT_DIR / 'corpus_fpr_report.csv'

# ---------- SETTINGS ----------
SEED = 37
AI_LABEL = 1
PRE_PERIOD = 'pre'
MODELS = os.environ.get('CORPUS_FPR_MODEL', 'tfidf').split(',')
# The boundary the shipped classifiers use, and the rate to calibrate against instead.
GLOBAL_BOUNDARY = 0.5
FPR_TARGET = 0.05
# Held identical to classify/distil.py so the two rates are comparable.
ENCODER = 'distilbert-base-uncased'
MAX_TOKENS = 256
BATCH_SIZE = 16
# Columns read from the corpus. body_text is the only large one; nothing else is needed.
CORPUS_COLUMNS = ('article_id', 'outlet', 'period', 'word_count', 'body_text')
