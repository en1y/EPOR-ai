"""A :class:`~epor.evaluation.scoring.ScoringModel` over an ``EporDecoder`` and a tokenizer.

Conventions
-----------
* Context and continuation are encoded separately and concatenated, so the
  continuation's tokens never depend on the context (and the length
  normalization denominators are stable); a prefix (``prefix_ids``) is added
  first: ``DOC_START`` for kernel checkpoints, whose training documents are
  framed ``DOC_START … DOC_END``, or ``BOS`` for the debug byte tokenizer.
* Sequences longer than ``max_length`` are truncated from the left of the
  context; a continuation that alone does not fit is an error.
* Batches are right-padded.  The decoder is causal, so padding after a
  sequence cannot change its logits; batch composition can still change
  floating-point rounding, so ``batch_size`` is part of the recorded identity.
* Generation is greedy (argmax, lowest ID on ties), restricted to IDs the
  tokenizer can decode plus the stop IDs, and stops at a stop ID, a stop
  string, ``max_new_tokens``, or ``max_length``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as F

from epor.models.reference import EporDecoder

from .report import ModelIdentity
from .scoring import Loglikelihood, ScoringModel, truncate_at_stop


class TextTokenizer(Protocol):
    def encode(self, value: str) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...


class DecoderScoringModel(ScoringModel):
    def __init__(
        self,
        model: EporDecoder,
        tokenizer: TextTokenizer,
        identity: ModelIdentity,
        *,
        prefix_ids: Sequence[int],
        stop_ids: Sequence[int] = (),
        decodable_vocab_size: int | None = None,
        max_length: int | None = None,
        batch_size: int = 8,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.prefix_ids = list(prefix_ids)
        self.stop_ids = list(stop_ids)
        self.max_length = min(
            max_length or model.config.configured_max_context,
            model.config.configured_max_context,
        )
        self.decodable_vocab_size = min(
            decodable_vocab_size or model.config.vocab_size, model.config.vocab_size
        )
        self.batch_size = batch_size
        self._identity = identity.model_copy(
            update={
                "scoring": {
                    **identity.scoring,
                    "batch_size": batch_size,
                    "max_length": self.max_length,
                    "prefix": ",".join(str(i) for i in self.prefix_ids),
                    "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
                }
            }
        )

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    def _sequence(self, context: str, continuation: str) -> tuple[list[int], int]:
        continuation_ids = self.tokenizer.encode(continuation)
        if not continuation_ids:
            raise ValueError("continuation must encode to at least one token")
        context_ids = self.prefix_ids + self.tokenizer.encode(context)
        if not context_ids:
            raise ValueError("context plus prefix must encode to at least one token")
        if len(continuation_ids) + 1 > self.max_length:
            raise ValueError("continuation does not fit in the scoring window")
        # The model input is ids[:-1]; keep at most max_length inputs.
        keep = self.max_length + 1 - len(continuation_ids)
        context_ids = context_ids[-keep:]
        return context_ids + continuation_ids, len(continuation_ids)

    def loglikelihood(self, context: str, continuation: str) -> Loglikelihood:
        return self.loglikelihood_batch([(context, continuation)])[0]

    @torch.no_grad()
    def loglikelihood_batch(self, requests: Sequence[tuple[str, str]]) -> list[Loglikelihood]:
        prepared = [self._sequence(context, continuation) for context, continuation in requests]
        results: list[Loglikelihood] = []
        for start in range(0, len(prepared), self.batch_size):
            chunk = prepared[start : start + self.batch_size]
            width = max(len(ids) - 1 for ids, _ in chunk)
            inputs = torch.zeros((len(chunk), width), dtype=torch.long)
            for row, (ids, _) in enumerate(chunk):
                inputs[row, : len(ids) - 1] = torch.tensor(ids[:-1], dtype=torch.long)
            logprobs = F.log_softmax(self.model(inputs).logits.float(), dim=-1)
            for row, (ids, count) in enumerate(chunk):
                positions = torch.arange(len(ids) - 1 - count, len(ids) - 1)
                targets = torch.tensor(ids[-count:], dtype=torch.long)
                selected = logprobs[row, positions]
                total = float(selected.gather(1, targets[:, None]).sum())
                greedy = bool((selected.argmax(dim=-1) == targets).all())
                results.append(Loglikelihood(sum_logprob=total, is_greedy=greedy))
        return results

    @torch.no_grad()
    def generate(self, prompt: str, *, max_new_tokens: int, stop: Sequence[str] = ()) -> str:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        ids = self.prefix_ids + self.tokenizer.encode(prompt)
        if not ids:
            raise ValueError("prompt plus prefix must encode to at least one token")
        mask = torch.full((self.model.config.vocab_size,), float("-inf"))
        mask[: self.decodable_vocab_size] = 0.0
        for stop_id in self.stop_ids:
            mask[stop_id] = 0.0
        generated: list[int] = []
        for _ in range(max_new_tokens):
            window = (ids + generated)[-self.max_length :]
            logits = self.model(torch.tensor([window], dtype=torch.long)).logits[0, -1].float()
            next_id = int(torch.argmax(logits + mask))
            if next_id in self.stop_ids:
                break
            generated.append(next_id)
            text, stopped = truncate_at_stop(self.tokenizer.decode(generated), stop)
            if stopped:
                return text
        return truncate_at_stop(self.tokenizer.decode(generated), stop)[0]


class KernelScoringModel(DecoderScoringModel):
    """A kernel checkpoint with the tokenizer of the build it was trained on."""

    checkpoint_dir: Path
    data_root: Path
    tokenized_id: str
    seeds: dict[str, int]


def load_kernel_scoring_model(
    checkpoint: str | Path,
    data_root: str | Path,
    *,
    batch_size: int = 8,
    max_length: int | None = None,
) -> KernelScoringModel:
    """Load a verified kernel checkpoint (FP32, CPU) and its build's tokenizer."""

    from epor.models.config import model_config_from_mapping
    from epor.reference_policy import validate_kernel_model
    from epor.tokenizer import DOC_END_ID, DOC_START_ID, EOS_ID, load_tokenizer
    from epor.tokenizer.shards import load_tokenized_manifest
    from epor.training.distributed import DistributedContext
    from epor.training.kernel import PackedTrainingSettings
    from epor.training.run_checkpoint import (
        MANIFEST_FILE,
        load_kernel_checkpoint,
        read_kernel_checkpoint,
    )

    trainer = read_kernel_checkpoint(checkpoint)
    directory = Path(trainer["checkpoint_dir"])
    config = validate_kernel_model(model_config_from_mapping(trainer["model_config"]))
    settings = PackedTrainingSettings.model_validate(trainer["settings"])
    root = Path(data_root).expanduser().resolve()
    manifest = load_tokenized_manifest(root, settings.tokenized_id)
    tokenizer = load_tokenizer(root, manifest.tokenizer_id)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("checkpoint vocabulary does not match its build's tokenizer")
    model = EporDecoder(config, seed=0, device="cpu", dtype=torch.float32, initialize=False)
    load_kernel_checkpoint(
        directory, model=model, optimizer=None, context=DistributedContext(), restore_rng=False
    )
    identity = ModelIdentity(
        kind="kernel-checkpoint",
        model_id=config.slug,
        config_sha256=config.sha256,
        checkpoint_sha256=hashlib.sha256((directory / MANIFEST_FILE).read_bytes()).hexdigest(),
        checkpoint_step=int(trainer["step"]),
        parameter_count=sum(p.numel() for p in model.parameters()),
        tokenizer_id=tokenizer.tokenizer_id,
        tokenizer_kind="epor-byte-bpe",
    )
    scoring = KernelScoringModel(
        model,
        tokenizer,
        identity,
        prefix_ids=[DOC_START_ID],
        stop_ids=[DOC_END_ID, EOS_ID],
        decodable_vocab_size=tokenizer.learned_vocab_size,
        max_length=max_length,
        batch_size=batch_size,
    )
    scoring.checkpoint_dir = directory
    scoring.data_root = root
    scoring.tokenized_id = settings.tokenized_id
    scoring.seeds = {"model": settings.model_seed, "data": settings.data_seed}
    return scoring


def debug_byte_scoring_model(
    model: EporDecoder,
    *,
    model_id: str = "in-memory-decoder",
    batch_size: int = 8,
) -> DecoderScoringModel:
    """An in-memory decoder with the v0.0.1 debug byte tokenizer (fast test path)."""

    from epor.models.tokenizer import DebugByteTokenizer

    tokenizer = DebugByteTokenizer()
    if model.config.vocab_size < tokenizer.vocab_size:
        raise ValueError("model vocabulary is smaller than the debug byte tokenizer")
    identity = ModelIdentity(
        kind="in-memory-decoder",
        model_id=model_id,
        config_sha256=model.config.sha256,
        parameter_count=sum(p.numel() for p in model.parameters()),
        tokenizer_id=tokenizer.sha256,
        tokenizer_kind="debug-byte-v1",
    )
    return DecoderScoringModel(
        model,
        tokenizer,
        identity,
        prefix_ids=[tokenizer.BOS_ID],
        stop_ids=[tokenizer.EOS_ID],
        decodable_vocab_size=tokenizer.vocab_size,
        batch_size=batch_size,
    )


__all__ = [
    "DecoderScoringModel",
    "KernelScoringModel",
    "TextTokenizer",
    "debug_byte_scoring_model",
    "load_kernel_scoring_model",
]
