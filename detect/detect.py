from __future__ import annotations

import csv
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.nn import functional
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from pathlib import Path

PERPLEXITY_MODEL = 'Qwen/Qwen2.5-0.5B'
RADAR_MODEL = 'TrustSafeAI/RADAR-Vicuna-7B'
RADAR_AI_INDEX = 0

LM_MAX_TOKENS = 1024
RADAR_MAX_TOKENS = 512


def select_device() -> str:
    """Return the best available torch device.

    Returns:
        str: Best available calculation device.

    """
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


# AI suggestion for Mac
def _dtype_for(device: str) -> torch.dtype:
    """Return bf16 on accelerators, fp32 on CPU."""
    return torch.bfloat16 if device in ('mps', 'cuda') else torch.float32


def _load_causal(model_id: str, device: str) -> tuple:
    """Load a causal LM + tokenizer in eval mode on device.

    Args:
        model_id (str): Model id.
        device (str): Target torch device.

    Returns:
        tuple: (tokenizer, model).

    """
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=_dtype_for(device))
    model.to(device).eval()
    return tok, model


class Detector:
    """Base detector: subclasses implement per-text scoring (higher = more AI)."""

    def __init__(
        self,
        name: str = 'base',
        max_tokens: int = LM_MAX_TOKENS,
        device: str | None = None,
    ) -> None:
        """Define base values for defining detector classes.

        Args:
            name (str, optional): Name of detector. Defaults to 'base'.
            max_tokens (int, optional): Max tokens for compute. Defaults to LM_MAX_TOKENS.
            device (str | None, optional): Device to do computation on. Defaults to None.

        """
        self.name: str = name
        self.max_tokens: int = max_tokens
        self.device = device or select_device()

    def score(self, texts: Sequence[str]) -> np.ndarray:
        """Score a list of texts; higher = more likely AI.

        Args:
            texts (Sequence[str]): Article bodies.

        Returns:
            np.ndarray: One float per text (higher = more AI).

        """
        raise NotImplementedError


class Perplexity(Detector):
    """Mean token log-probability under a small LM (higher = more AI)."""

    name = 'perplexity'

    def __init__(
        self, model_id: str = PERPLEXITY_MODEL, device: str | None = None
    ) -> None:
        """Load the reference LM.

        Args:
            model_id (str): HF model id for the reference LM.
            device (str | None): Torch device; auto-selected if None.

        """
        super().__init__(
            name=self.name,
            device=device,
        )

        self.tok, self.model = _load_causal(model_id, self.device)


class Radar(Detector):
    """Supervised RoBERTa detector (RADAR); returns P(text is AI-generated)."""

    name = 'radar'
    max_tokens = RADAR_MAX_TOKENS

    def __init__(
        self,
        model_id: str = RADAR_MODEL,
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        """Load the RADAR sequence-classification detector.

        Args:
            model_id (str): Id of the RADAR detector.
            device (str | None): Torch device; auto-selected if None.
            batch_size (int): Sequences per forward pass.

        """
        super().__init__(
            name=self.name,
            max_tokens=self.max_tokens,
            device=device,
        )
        self.batch_size = batch_size
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.to(device).eval()

    @torch.no_grad()
    def score(self, texts: Sequence[str]) -> np.ndarray:
        """Score a list of texts; higher = more likely AI.

        Args:
            texts (Sequence[str]): Article bodies.

        Returns:
            np.ndarray: One float per text (higher = more AI).

        """
        output = np.empty(len(texts), dtype=np.float64)
        for start in range(0, len(texts), self.batch_size):
            chunk = list(texts[start : start + self.batch_size])
            enc = self.tok(
                chunk,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=self.max_tokens,
            ).to(self.device)
            probs = functional.softmax(self.model(**enc).logits.float(), dim=-1)
            output[start : start + len(chunk)] = probs[:, RADAR_AI_INDEX].cpu().numpy()
        return output


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
