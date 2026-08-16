"""Score the fine-tuned classifier: held-out F1, per-generator recall, corpus FPR."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
SPLIT_DIR = DATA / 'classify_irish'
MODEL_DIR = ROOT / 'models' / 'deberta-v3-base
GENERATED = DATA / 'generation' / 'generated_irish_ai.csv'
REPORT = SPLIT_DIR / 'classifier_report.csv'

MAX_TOKENS = 512
BATCH_SIZE = 32
AI_LABEL = 1
DECISION_THRESHOLD = 0.5
PROPOSAL_TARGET_F1 = 0.85


def predict(texts: list[str], model_dir: Path = MODEL_DIR) -> np.ndarray:
    """Return P(AI) for each text.

    Args:
        texts (list[str]): Documents to score.
        model_dir (Path): Directory holding the fine-tuned weights.

    Returns:
        np.ndarray: Probability of the AI class, one per text.

    Raises:
        FileNotFoundError: If the model directory does not exist.

    """


    if not model_dir.exists():
        message = f'{model_dir} not found; run classify.train_irish_stream first.'
        raise FileNotFoundError(message)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()

    scored = []
    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            encoded = tokenizer(
                texts[start : start + BATCH_SIZE],
                truncation=True,
                max_length=MAX_TOKENS,
                padding=True,
                return_tensors='pt',
            ).to(device)
            logits = model(**encoded).logits
            scored.append(torch.softmax(logits, dim=-1)[:, AI_LABEL].cpu().numpy())
    return np.concatenate(scored)
def evaluate() -> pd.DataFrame:
    """Return overall metrics plus recall per generator and per era.

    Returns:
        pd.DataFrame: One row per scope.

    """

    test = pd.read_csv(SPLIT_DIR / 'test.csv')
    logger.info('scoring %d held-out documents', len(test))
    test['predicted'] = (
        predict(test['text'].astype(str).tolist()) >= DECISION_THRESHOLD
    ).astype(int)

    precision, recall, f1, _unused = precision_recall_fscore_support(
        test['label'],
        test['predicted'],
        average='binary',
        pos_label=AI_LABEL,
        zero_division=0,
    )
    records = [
        {
            'scope': 'overall',
            'n': len(test),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'meets_target': bool(f1 >= PROPOSAL_TARGET_F1),
        }
    ]

    ai_rows = test[test['label'] == AI_LABEL]
    groups = {f'recall:{name}': rows for name, rows in ai_rows.groupby('model')}
    if GENERATED.exists():
        eras = pd.read_csv(GENERATED, usecols=['id', 'era']).set_index('id')['era']
        with_era = ai_rows.assign(era=ai_rows['id'].map(eras)).dropna(subset=['era'])
        groups |= {
            f'recall_era:{int(era)}': rows for era, rows in with_era.groupby('era')
        }
    for scope, rows in groups.items():
        records.append(
            {
                'scope': scope,
                'n': len(rows),
                'precision': float('nan'),
                'recall': float((rows['predicted'] == AI_LABEL).mean()),
                'f1': float('nan'),
                'meets_target': False,
            }
        )

    anchor = test[test['label'] != AI_LABEL]
    records.append(
        {
            'scope': 'false_positive_rate:anchor',
            'n': len(anchor),
            'precision': float('nan'),
            'recall': float((anchor['predicted'] == AI_LABEL).mean()),
            'f1': float('nan'),
            'meets_target': False,
        }
    )
    return pd.DataFrame(records)

def main() -> int:
    """Evaluate the Irish classifier and write its report.

    Returns:
        int: 0 on success, 1 if a required input is missing.

    """
    try:
        report = evaluate()
    except FileNotFoundError:
        logger.exception('missing input')
        return 1

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT, index=False)
    overall = report[report['scope'] == 'overall'].iloc[0]
    logger.info(
        'held-out F1 = %.4f (target %.2f) -> %s',
        overall['f1'],
        PROPOSAL_TARGET_F1,
        'MET' if overall['meets_target'] else 'NOT MET',
    )
    for row in report[report['scope'].str.startswith('recall_era:')].itertuples():
        logger.info('  %-22s %.4f', row.scope, row.recall)
    logger.info('wrote %s', REPORT.name)
    return 0


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )
    sys.exit(main())
