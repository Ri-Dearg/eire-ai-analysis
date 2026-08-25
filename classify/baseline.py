"""Linear baseline for the Irish classifier: TF-IDF over word and character n-grams."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import make_pipeline

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT / 'data' / 'classify'
REPORT = SPLIT_DIR / 'baseline_report.csv'
AI_LABEL = 1
SEED = 37


def main() -> int:
    """Fit the baseline and write its per-generator report.

    Returns:
        int: 0 on success, 1 if the splits are missing.

    """
    if not (SPLIT_DIR / 'train.csv').exists():
        logger.error('no splits at %s; run classify.dataset', SPLIT_DIR)
        return 1

    train = pd.read_csv(SPLIT_DIR / 'train.csv')
    test = pd.read_csv(SPLIT_DIR / 'test.csv')

    model = make_pipeline(
        TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=2000, random_state=SEED),
    ).fit(train['text'].astype(str), train['label'])

    test['predicted'] = model.predict(test['text'].astype(str))
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

    ai = test[test['label'] == AI_LABEL]
    for name, group in ai.groupby('model'):
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
    logger.info('wrote %s', REPORT.name)
    return 0


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )
    sys.exit(main())
