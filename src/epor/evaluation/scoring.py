"""The model interface evaluation tasks score against (no torch import here).

Concrete adapters live in :mod:`epor.evaluation.decoder` (an
:class:`~epor.models.reference.EporDecoder` plus a tokenizer, including kernel
checkpoints); tests may implement :class:`ScoringModel` with fixed outputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from .report import ModelIdentity


@dataclass(frozen=True, slots=True)
class Loglikelihood:
    """Summed log-probability of a continuation and whether greedy decoding yields it."""

    sum_logprob: float
    is_greedy: bool


class ScoringModel(ABC):
    """Deterministic scoring surface: continuation log-likelihoods and greedy generation."""

    @property
    @abstractmethod
    def identity(self) -> ModelIdentity:
        """Identity recorded in reports (config/checkpoint hashes, tokenizer)."""

    @abstractmethod
    def loglikelihood(self, context: str, continuation: str) -> Loglikelihood:
        """``log P(continuation | context)`` summed over the continuation's tokens."""

    def loglikelihood_batch(self, requests: Sequence[tuple[str, str]]) -> list[Loglikelihood]:
        """Score many ``(context, continuation)`` pairs; adapters may batch."""

        return [self.loglikelihood(context, continuation) for context, continuation in requests]

    @abstractmethod
    def generate(self, prompt: str, *, max_new_tokens: int, stop: Sequence[str] = ()) -> str:
        """Greedy continuation of ``prompt``, cut before the earliest ``stop`` string."""


def truncate_at_stop(text: str, stop: Sequence[str]) -> tuple[str, bool]:
    """Cut ``text`` before the earliest occurrence of any non-empty stop string."""

    cut = len(text)
    for marker in stop:
        if marker:
            index = text.find(marker)
            if 0 <= index < cut:
                cut = index
    return text[:cut], cut < len(text)


__all__ = ["Loglikelihood", "ScoringModel", "truncate_at_stop"]
