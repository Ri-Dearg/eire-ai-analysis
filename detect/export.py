from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
CORPUS = DATA / 'corpus.csv'
DET_DIR = DATA / 'detection'
DEFAULT_OUT = DATA / 'corpus_scored.csv'

DETECTORS = ('perplexity', 'radar', 'binoculars', 'fastdetectgpt')

CATEGORY = {
    'rte': 'legacy',
    'irish_examiner': 'legacy',
    'the_liberal': 'counter-consensus',
    'gript': 'counter-consensus',
}

META_COLS = [
    'article_id',
    'url_canonical',
    'outlet',
    'published_date',
    'year',
    'month',
    'period',
    'section',
    'author',
    'is_wire',
    'word_count',
]
OUT_COLS = [*META_COLS, 'category', *(f'{detector}_score' for detector in DETECTORS)]

csv.field_size_limit(1 << 24)


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


def export(output_path: Path) -> dict[str, int]:
    """Join corpus metadata with every detector checkpoint and write output.

    Args:
        output_path (Path): Destination CSV.

    Returns:
        dict[str, int]: Rows written plus per-detector coverage counts.

    """
    checkpoints = {detector: load_checkpoint(detector) for detector in DETECTORS}
    counts = dict.fromkeys(DETECTORS, 0)
    written = 0
    with (
        CORPUS.open(encoding='utf-8') as src,
        output_path.open('w', newline='', encoding='utf-8') as score_directory,
    ):
        reader = csv.DictReader(src)
        writer = csv.DictWriter(
            score_directory, fieldnames=OUT_COLS, extrasaction='ignore'
        )
        writer.writeheader()
        for row in reader:
            output = {col: row.get(col, '') for col in META_COLS}
            output['category'] = CATEGORY.get(row['outlet'], '')
            for detector in DETECTORS:
                score = checkpoints[detector].get(row['article_id'], '')
                output[f'{detector}_score'] = score
                counts[detector] += bool(score)
            writer.writerow(output)
            written += 1
    return {'rows': written, **counts}
