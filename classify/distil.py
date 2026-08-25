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


def embed(texts: list[str], tokenizer: object, model: object) -> np.ndarray:
    """Return mean-pooled encoder embeddings, masked over real tokens only.

    Args:
        texts (list[str]): Documents to encode.
        tokenizer (object): Matching tokenizer.
        model (object): Frozen encoder.

    Returns:
        np.ndarray: One embedding row per document.

    """
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch = tokenizer(
                texts[start : start + BATCH_SIZE],
                truncation=True,
                max_length=MAX_TOKENS,
                padding=True,
                return_tensors='pt',
            )
            hidden = model(**batch).last_hidden_state
            mask = batch['attention_mask'].unsqueeze(-1).float()
            out.append(((hidden * mask).sum(1) / mask.sum(1)).numpy())
            if start % (BATCH_SIZE * 20) == 0:
                logger.info('  encoded %d/%d', start, len(texts))
    return np.vstack(out)


def main() -> int:
    """Fit the frozen-encoder classifier and write its report.

    Returns:
        int: 0 on success, 1 if the splits are missing.

    """
    if not (SPLIT_DIR / 'train.csv').exists():
        logger.error('no splits at %s; run classify.dataset', SPLIT_DIR)
        return 1

    tokenizer = AutoTokenizer.from_pretrained(ENCODER)
    model = AutoModel.from_pretrained(ENCODER).eval()  # CPU: inference only, no optimiser

    train = pd.read_csv(SPLIT_DIR / 'train.csv')
    test = pd.read_csv(SPLIT_DIR / 'test.csv')
    logger.info('encoding %d train', len(train))
    x_train = embed(train['text'].astype(str).tolist(), tokenizer, model)
    logger.info('encoding %d test', len(test))
    x_test = embed(test['text'].astype(str).tolist(), tokenizer, model)

    head = LogisticRegression(max_iter=2000, random_state=SEED).fit(
        x_train, train['label']
    )
    test['predicted'] = head.predict(x_test)

    precision, recall, f1, _unused = precision_recall_fscore_support(
        test['label'],
        test['predicted'],
        average='binary',
        pos_label=AI_LABEL,
        zero_division=0,
    )
    rows = [
        {
            'scope': 'overall',
            'n': len(test),
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    ]
    for name, group in test[test['label'] == AI_LABEL].groupby('model'):
        rows.append(
            {
                'scope': f'recall:{name}',
                'n': len(group),
                'precision': float('nan'),
                'recall': float((group['predicted'] == AI_LABEL).mean()),
                'f1': float('nan'),
            }
        )
    human = test[test['label'] != AI_LABEL]
    rows.append(
        {
            'scope': 'false_positive_rate:anchor',
            'n': len(human),
            'precision': float('nan'),
            'recall': float((human['predicted'] == AI_LABEL).mean()),
            'f1': float('nan'),
        }
    )

    report = pd.DataFrame(rows)
    report.to_csv(REPORT, index=False)
    logger.info('held-out F1 = %.4f', f1)
    for row in report[report['scope'].str.startswith('recall:')].itertuples():
        logger.info('  %-28s %.4f  (n=%d)', row.scope, row.recall, row.n)
    return 0


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )
    sys.exit(main())
