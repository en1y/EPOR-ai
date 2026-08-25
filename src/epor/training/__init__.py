"""Public reference-training API.

PyTorch is an optional ``train-cpu`` dependency, so callers should import this
module only for model execution commands.  Configuration-only validation remains
available from :mod:`epor.models` in the default environment.
"""

from __future__ import annotations

from pathlib import Path

from epor.models.config import load_model_config, validate_config

from .checkpoint import load_checkpoint
from .data import DEFAULT_FIXTURE_CORPUS, RandomBatchStream, load_token_corpus
from .loop import (
    EvaluationResult,
    GenerationResult,
    TrainingCancelled,
    TrainingResult,
    TrainingSettings,
    evaluate,
    generate,
    pretrain,
)


def resume_pretraining(
    config_path: str,
    output_dir: str,
    checkpoint_path: str,
    *,
    max_steps: int,
    corpus_path: str | Path = DEFAULT_FIXTURE_CORPUS,
) -> TrainingResult:
    """Resume to the absolute ``max_steps`` using saved trajectory settings."""

    return pretrain(
        config_path,
        output_dir,
        max_steps=max_steps,
        resume_from=checkpoint_path,
        corpus_path=corpus_path,
    )


__all__ = [
    "DEFAULT_FIXTURE_CORPUS",
    "EvaluationResult",
    "GenerationResult",
    "RandomBatchStream",
    "TrainingCancelled",
    "TrainingResult",
    "TrainingSettings",
    "evaluate",
    "generate",
    "load_checkpoint",
    "load_model_config",
    "load_token_corpus",
    "pretrain",
    "resume_pretraining",
    "validate_config",
]
