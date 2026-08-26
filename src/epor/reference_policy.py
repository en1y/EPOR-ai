"""Shared resource policy for the bounded v0.0.1 reference runtime.

The reference trainer is intentionally a correctness harness, not a route to
full-scale training. These checks stay free of PyTorch so both the base control
plane and optional execution environment can enforce the same contract before
allocating model or corpus state.
"""

from __future__ import annotations

from pathlib import Path

from epor.models.config import ModelConfig, load_model_config, structural_parameter_count

REFERENCE_CORPUS_MAX_BYTES = 1 * 1024 * 1024
REFERENCE_CHECKPOINT_MAX_BYTES = 768 * 1024 * 1024
REFERENCE_MODEL_MAX_PARAMETERS = 50_000_000
REFERENCE_MAX_STEPS = 1_000
REFERENCE_MAX_EVAL_BATCHES = 10_000
REFERENCE_MAX_BATCH_TOKENS = 4_096
REFERENCE_MAX_NEW_TOKENS = 4_096


def validate_bounded_file(
    path: str | Path,
    *,
    label: str,
    max_bytes: int,
) -> Path:
    """Resolve a regular file and reject it before loading when it is too large."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    size = resolved.stat().st_size
    if size > max_bytes:
        raise ValueError(f"{label} exceeds the {max_bytes:,}-byte reference limit")
    return resolved


def validate_reference_corpus(path: str | Path) -> Path:
    """Validate the in-memory fixture-corpus boundary before reading bytes."""

    return validate_bounded_file(
        path,
        label="corpus",
        max_bytes=REFERENCE_CORPUS_MAX_BYTES,
    )


def validate_reference_checkpoint(path: str | Path) -> Path:
    """Validate the trusted checkpoint container before ``torch.load`` allocates it."""

    return validate_bounded_file(
        path,
        label="checkpoint",
        max_bytes=REFERENCE_CHECKPOINT_MAX_BYTES,
    )


def validate_reference_model(config: ModelConfig) -> ModelConfig:
    """Reject target-family or oversized recipes before allocating a decoder."""

    if config.architecture != "dense-decoder" or config.vocab_size != 260:
        raise ValueError("the v0.0.1 reference runtime requires a dense 260-token model")
    parameters = structural_parameter_count(config)
    if parameters > REFERENCE_MODEL_MAX_PARAMETERS:
        raise ValueError("reference model exceeds the 50,000,000-parameter execution limit")
    return config


def load_reference_model_config(path: str | Path) -> ModelConfig:
    """Load and validate a recipe under the executable reference-model contract."""

    return validate_reference_model(load_model_config(path))


def validate_reference_steps(max_steps: int) -> int:
    if not 1 <= max_steps <= REFERENCE_MAX_STEPS:
        raise ValueError(f"max_steps must be between 1 and {REFERENCE_MAX_STEPS}")
    return max_steps


def validate_reference_eval_batches(max_batches: int | None) -> int | None:
    if max_batches is not None and not 1 <= max_batches <= REFERENCE_MAX_EVAL_BATCHES:
        raise ValueError(
            f"max_batches must be between 1 and {REFERENCE_MAX_EVAL_BATCHES} when provided"
        )
    return max_batches


def validate_reference_batch(batch_size: int, sequence_length: int) -> None:
    if batch_size < 1 or sequence_length < 1:
        raise ValueError("batch_size and sequence_length must be positive")
    tokens = batch_size * sequence_length
    if tokens > REFERENCE_MAX_BATCH_TOKENS:
        raise ValueError(
            f"batch_size * sequence_length exceeds the {REFERENCE_MAX_BATCH_TOKENS:,}-token "
            "reference limit"
        )


def validate_reference_generation(max_new_tokens: int) -> int:
    if not 0 <= max_new_tokens <= REFERENCE_MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens must be between 0 and {REFERENCE_MAX_NEW_TOKENS}")
    return max_new_tokens
