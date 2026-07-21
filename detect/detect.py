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

MIN_TOKENS = 2
LM_MAX_TOKENS = 1024
RADAR_MAX_TOKENS = 512
DET_BATCH = 1
DET_CHUNK = 32


# AI suggested batching
def _batched_map(
    texts: Sequence[str], batch_size: int, per_batch: Callable[[list[str]], np.ndarray]
) -> np.ndarray:
    """Score texts in length-sorted minibatches, restoring input order.

    Args:
        texts (Sequence[str]): All documents to score.
        batch_size (int): Documents per pass.
        per_batch (Callable[[list[str]], np.ndarray]): Scores one minibatch,
            returning one float per document.

    Returns:
        np.ndarray: One float per input document, aligned to ``texts``.

    """
    order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
    output = np.empty(len(texts), dtype=np.float64)
    for start in range(0, len(order), batch_size):
        indexes = order[start : start + batch_size]
        scores = per_batch([texts[index] for index in indexes])
        for position, src in enumerate(indexes):
            output[src] = scores[position]
    return output


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of values over real positions, NaN where a row has none.

    Args:
        values (torch.Tensor): Per-position values, shape (batch, length).
        mask (torch.Tensor): 1.0 for real positions, 0.0 for padding, same shape.

    Returns:
        torch.Tensor: Per-row mean, shape, NaN for empty rows.

    """
    counts = mask.sum(1)
    total = (values * mask).sum(1)
    out = total / counts.clamp_min(1.0)
    out[counts < 1.0] = float('nan')
    return out


# AI Suggested batch padding
def _pad_batch(
    tokeniser: object, texts: Sequence[str], max_tokens: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize texts into a right-padded, truncated batch on device.

    Right padding keeps every real token's causal context intact, so the per-token
    log-probabilities of the real tokens are unaffected by the pad positions.

    Args:
        tok (object): A Hugging Face tokenizer.
        texts (Sequence[str]): Article bodies.
        max_tokens (int): Truncation window.
        device (str): Torch device string.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (input_ids, attention_mask), each
        shape (batch, seq).

    """
    tokeniser.padding_side = 'right'
    if tokeniser.pad_token_id is None:
        tokeniser.pad_token = tokeniser.eos_token
    encode = tokeniser(
        list(texts),
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_tokens,
    )
    return encode['input_ids'].to(device), encode['attention_mask'].to(device)


def _target_logprobs(shift_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Log P(target) per position,.

    Args:
        shift_logits (torch.Tensor): Logits for predicted positions.
        targets (torch.Tensor): Observed next-token ids.

    Returns:
        torch.Tensor: Per-position target log-probabilities.

    """
    batch_size, length, _vocab = shift_logits.shape
    output = torch.empty(
        (batch_size, length), dtype=torch.float32, device=shift_logits.device
    )
    for chunk_start in range(0, length, DET_CHUNK):
        chunk_end = min(chunk_start + DET_CHUNK, length)
        chunk = shift_logits[:, chunk_start:chunk_end, :].float()
        lse = torch.logsumexp(chunk, dim=-1)
        gathered = chunk.gather(
            -1, targets[:, chunk_start:chunk_end].unsqueeze(-1)
        ).squeeze(-1)
        output[:, chunk_start:chunk_end] = gathered - lse
    return output


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
    tokeniser = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=_dtype_for(device))
    model.to(device).eval()
    return tokeniser, model


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
            model_id (str): Model id for the reference LM.
            device (str | None): Torch device; auto-selected if None.

        """
        super().__init__(
            name=self.name,
            device=device,
        )

        self.tokeniser, self.model = _load_causal(model_id, self.device)

    def score(self, texts: Sequence[str]) -> np.ndarray:
        """Score a list of texts; higher = more likely AI.

        Args:
            texts (Sequence[str]): Article bodies.

        Returns:
            np.ndarray: One float per text (higher = more AI).

        """
        return _batched_map(texts, DET_BATCH, self._score_batch)

    @torch.no_grad()
    def _score_batch(self, texts: list[str]) -> np.ndarray:
        """Score one minibatch; higher = more AI."""
        ids, attn = _pad_batch(self.tokeniser, texts, self.max_tokens, self.device)
        if ids.shape[1] < MIN_TOKENS:
            return np.full(len(texts), np.nan)
        logits = self.model(ids, attention_mask=attn).logits
        log_probs = _target_logprobs(logits[:, :-1, :], ids[:, 1:])
        return _masked_mean(log_probs, attn[:, 1:].float()).cpu().numpy()


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
        self.tokeniser = AutoTokenizer.from_pretrained(model_id)
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
            enc = self.tokeniser(
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
