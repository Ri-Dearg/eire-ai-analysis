"""Frozen-encoder classifier: DistilBERT embeddings + logistic head."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT / 'data' / 'classify'
REPORT = SPLIT_DIR / 'frozen_report.csv'
ENCODER = 'distilbert-base-uncased'
MAX_TOKENS = 256
BATCH_SIZE = 16
AI_LABEL = 1
SEED = 37
