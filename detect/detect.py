from __future__ import annotations

import csv
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from pathlib import Path


def run_detector(
    detector,
    ids: Sequence[str],
    texts: Sequence[str],
    output_path: Path,
    batch: int = 64,
) -> Path:
    """Score (id, text) pairs with checkpointing; resumable across restarts.

    Args:
        detector (Detector): A constructed detector.
        ids (Sequence[str]): Stable ids aligned with texts.
        texts (Sequence[str]): Texts to score.
        output_path (Path): Checkpoint CSV path.
        batch (int): Texts scored (and flushed) per checkpoint write.

    Returns:
        Path: output_path.

    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not output_path.exists()
    pending = list(zip(ids, texts, strict=True))
    with output_path.open('a', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(['id', f'{detector.name}_score'])
        for start in range(0, len(pending), batch):
            chunk = pending[start : start + batch]
            scores = detector.score([text for _, text in chunk])
            writer.writerows(
                [
                    (cid, float(score))
                    for (cid, _), score in zip(chunk, scores, strict=True)
                ]
            )
            fh.flush()
    return output_path
