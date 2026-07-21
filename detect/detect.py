from __future__ import annotations

import csv
import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from pathlib import Path

LM_MAX_TOKENS = int('1024')


class Detector:
    """Base detector: subclasses implement per-text scoring (higher = more AI)."""

    name: str = 'base'
    max_tokens: int = LM_MAX_TOKENS

    def score(self, texts: Sequence[str]) -> np.ndarray:
        """Score a list of texts; higher = more likely AI.

        Args:
            texts (Sequence[str]): Article bodies.

        Returns:
            np.ndarray: One float per text (higher = more AI).

        """
        raise NotImplementedError


# Resumable design by AI
def _done_ids(path: Path) -> set[str]:
    """Return ids already scored in a checkpoint CSV (empty if absent).

    Args:
        path (Path): Path to scored article file.

    Returns:
        set[str]: Set of scored article rows.

    """
    if not path.exists():
        return set()
    with path.open(encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader, None)
        return {row[0] for row in reader if row}


def run_detector(
    detector: Detector,
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
    done = _done_ids(output_path)
    new_file = not output_path.exists()
    pending = [
        (index, text) for index, text in zip(ids, texts, strict=True) if index not in done
    ]
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
