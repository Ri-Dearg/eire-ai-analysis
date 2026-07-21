"""Per-outlet calibration, corpus scoring, and output tables for the detectors.

Turns raw detector scores into a measured lower bound on AI.
"""

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
    from collections.abc import Callable, Sequence
    from pathlib import Path

# ---------- MODEL SETTINGS ----------
BINOCULARS_OBSERVER = 'Qwen/Qwen2.5-1.5B'
BINOCULARS_PERFORMER = 'Qwen/Qwen2.5-1.5B-Instruct'
FASTDETECT_MODEL = 'Qwen/Qwen2.5-3B'
PERPLEXITY_MODEL = 'Qwen/Qwen2.5-0.5B'
_EPS = 1e-6
RADAR_MODEL = 'TrustSafeAI/RADAR-Vicuna-7B'
RADAR_AI_INDEX = 0
RADAR_MAX_TOKENS = 512

MIN_TOKENS = 2
LM_MAX_TOKENS = 1024
DET_BATCH = 1
DET_CHUNK = 32


# ---------- BATCHING ----------
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


# AI Suggested batch padding
def _pad_batch(
    tokeniser: object, texts: Sequence[str], max_tokens: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize texts into a right-padded, truncated batch on device.

    Right padding keeps every real token's causal context intact, so the per-token
    log-probabilities of the real tokens are unaffected by the pad positions.

    Args:
        tokeniser (object): A Hugging Face tokenizer.
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


# ---------- CALCULATIONS ----------
def _cross_perplexity(
    obs_logits: torch.Tensor,
    perf_logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Binoculars per-position terms.

    Args:
        obs_logits (torch.Tensor): Observer predicted-position logits.
        perf_logits (torch.Tensor): Performer predicted-position logits.
        targets (torch.Tensor): Observed next-token ids.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The observer log-prob of the observed token
        and the performer/observer cross`.

    """
    batch_size, length, _vocab = obs_logits.shape
    device = obs_logits.device
    obs_target = torch.empty((batch_size, length), dtype=torch.float32, device=device)
    cross = torch.empty((batch_size, length), dtype=torch.float32, device=device)
    for chunk_start in range(0, length, DET_CHUNK):
        chunk_end = min(chunk_start + DET_CHUNK, length)
        obs_logp = functional.log_softmax(
            obs_logits[:, chunk_start:chunk_end, :].float(), dim=-1
        )
        perf_p = functional.softmax(
            perf_logits[:, chunk_start:chunk_end, :].float(), dim=-1
        )
        obs_target[:, chunk_start:chunk_end] = obs_logp.gather(
            -1, targets[:, chunk_start:chunk_end].unsqueeze(-1)
        ).squeeze(-1)
        cross[:, chunk_start:chunk_end] = (perf_p * obs_logp).sum(-1)
    return obs_target, cross


def _curvature_stats(
    shift_logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Analytic Fast-DetectGPT statistics.

    Args:
        shift_logits (torch.Tensor): Predicted-position logits.
        targets (torch.Tensor): Next-token ids.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: the observed token log-prob
        and the mean and variance of the token prob under the model's own distribution.

    """
    batch_size, length, _vocab = shift_logits.shape
    device = shift_logits.device
    observed = torch.empty((batch_size, length), dtype=torch.float32, device=device)
    expect_mean = torch.empty((batch_size, length), dtype=torch.float32, device=device)
    variance = torch.empty((batch_size, length), dtype=torch.float32, device=device)
    for chunk_start in range(0, length, DET_CHUNK):
        chunk_end = min(chunk_start + DET_CHUNK, length)
        logp = functional.log_softmax(
            shift_logits[:, chunk_start:chunk_end, :].float(), dim=-1
        )
        prob = logp.exp()
        mean_curve = (prob * logp).sum(-1)
        observed[:, chunk_start:chunk_end] = logp.gather(
            -1, targets[:, chunk_start:chunk_end].unsqueeze(-1)
        ).squeeze(-1)
        expect_mean[:, chunk_start:chunk_end] = mean_curve
        variance[:, chunk_start:chunk_end] = (prob * logp.pow(2)).sum(
            -1
        ) - mean_curve.pow(2)
    return observed, expect_mean, variance


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


# ---------- DEVICE + TOKENS ----------
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
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=_dtype_for(device))
    model.to(device).eval()
    return tokeniser, model


# ---------- DETECTORS ----------
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


class Binoculars(Detector):
    """Observer log-perplexity / observer-performer cross-perplexity.

    Returns ``-B`` so higher = more AI (machine text has low Binoculars score).
    """

    name = 'binoculars'

    def __init__(
        self,
        observer_id: str = BINOCULARS_OBSERVER,
        performer_id: str = BINOCULARS_PERFORMER,
        device: str | None = None,
    ) -> None:
        """Load the observer and performer LMs.

        Args:
            observer_id (str): Id of the observer LM.
            performer_id (str): Id of the performer LM.
            device (str | None): Torch device; auto-selected if None.

        """
        super().__init__(
            name=self.name,
            device=device,
        )
        self.tokeniser, self.observer = _load_causal(observer_id, self.device)
        _, self.performer = _load_causal(performer_id, self.device)

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
        """Score one minibatch; returns ``-B`` so higher = more AI."""
        ids, attn = _pad_batch(self.tokeniser, texts, self.max_tokens, self.device)
        if ids.shape[1] < MIN_TOKENS:
            return np.full(len(texts), np.nan)
        obs_logits = self.observer(ids, attention_mask=attn).logits
        perf_logits = self.performer(ids, attention_mask=attn).logits
        obs_target, cross = _cross_perplexity(
            obs_logits[:, :-1, :], perf_logits[:, :-1, :], ids[:, 1:]
        )
        mask = attn[:, 1:].float()
        log_perplex = -_masked_mean(obs_target, mask)
        x_perplex = -_masked_mean(cross, mask)
        return (-(log_perplex / x_perplex.clamp_min(_EPS))).cpu().numpy()


class FastDetectGPT(Detector):
    """Sampling-free conditional-probability curvature."""

    name = 'fastdetectgpt'

    def __init__(
        self, model_id: str = FASTDETECT_MODEL, device: str | None = None
    ) -> None:
        """Load the scoring LM.

        Args:
            model_id (str): HF model id for the scoring LM.
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
        observed, expect_mean, var = _curvature_stats(logits[:, :-1, :], ids[:, 1:])
        mask = attn[:, 1:].float()
        num = ((observed - expect_mean) * mask).sum(1)
        denom = torch.sqrt((var * mask).sum(1)).clamp_min(_EPS)
        out = num / denom
        out[mask.sum(1) < 1.0] = float('nan')
        return out.cpu().numpy()


class Perplexity(Detector):
    """Mean token log-probability under a small LM."""

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
        self.model.to(self.device).eval()

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


DETECTORS = {
    'perplexity': Perplexity,
    'fastdetectgpt': FastDetectGPT,
    'binoculars': Binoculars,
    'radar': Radar,
}


def build_detector(name: str, **kwargs: object) -> Detector:
    """Instantiate a detector by name.

    Args:
        name (str): One of :data:`DETECTORS`.
        **kwargs (object): Passed to the detector constructor.

    Returns:
        Detector: The constructed detector.

    """
    return DETECTORS[name](**kwargs)


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
    with output_path.open('a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
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
            file.flush()
    return output_path
