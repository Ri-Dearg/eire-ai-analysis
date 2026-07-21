from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
CORPUS = DATA / 'corpus.csv'
DET_DIR = DATA / 'detection'
DEFAULT_OUT = DATA / 'corpus_v2_scored.csv'


def load_checkpoint(detector: str) -> dict[str, str]:
    """Return article_id -> score for one detector's corpus rows.

    Reads data/detection/<detector>.csv.

    Args:
        detector (str): Detector name.

    Returns:
        dict[str, str]: Scores keyed by bare article id; empty if the
            checkpoint file is absent.

    """
    path = DET_DIR / f'{detector}.csv'
    if not path.exists():
        return {}
    scores: dict[str, str] = {}
    with path.open(encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0].startswith('corpus:'):
                scores[row[0].removeprefix('corpus:')] = row[1]
    return scores
