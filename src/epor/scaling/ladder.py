"""The scaling-laboratory model-size ladder: 10M -> 100M -> 300M -> 1B.

Rungs are dense decoders in ``configs/models/ladder/``.  The rung size is the
**non-embedding** parameter count (Kaplan's ``N``): at the 65,536-entry
target vocabulary the tied embedding alone is 25M-134M parameters, so a
total-count ladder would mostly measure the vocabulary.  Each file also
declares its structural total.  Shapes follow the shipped recipes: 64-wide
heads up to 300M (128 at 1B, as in the β/γ recipes), ``d_ff = 3 d_model``,
grouped-query attention with 3:1 or 4:1 query-to-KV heads.  The files are
architecture contracts only; none of them is a trained or validated model.

Local execution is decided by the training kernel's existing bound
(:func:`epor.reference_policy.validate_kernel_model`, 50M structural
parameters).  At the target vocabulary only the 10M rung fits (35.5M total);
100M, 300M, and 1B are **projection-only** on the reference machine whatever
the vocabulary.  Local runs train a *derived* config with the tokenized
build's vocabulary, exactly as ``--vocab-from-build`` derives one
(:func:`derive_rung`); the rung file's hash never changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from epor.models.config import ModelConfig, load_model_config, structural_parameter_count
from epor.reference_policy import REFERENCE_MODEL_MAX_PARAMETERS

from .flops import parameter_counts

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LADDER_DIRECTORY = Path("configs/models/ladder")
LADDER_RUNGS: dict[str, int] = {
    "epor-ladder-10m": 10_000_000,
    "epor-ladder-100m": 100_000_000,
    "epor-ladder-300m": 300_000_000,
    "epor-ladder-1b": 1_000_000_000,
}
# Each rung's non-embedding count lands within this relative distance of its
# nominal size (tests pin it).
LADDER_TOLERANCE = 0.05

ExecutionStatus = Literal["local", "projection-only"]


class RungStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    slug: str
    config_sha256: str
    nominal_non_embedding: int | None
    total_parameters: int
    non_embedding_parameters: int
    vocab_size: int
    execution: ExecutionStatus
    reason: str


def ladder_paths(project_root: str | Path = PROJECT_ROOT) -> list[Path]:
    """The shipped rung files, smallest first."""

    root = Path(project_root) / LADDER_DIRECTORY
    return [root / f"{slug}.yaml" for slug in LADDER_RUNGS]


def load_ladder(project_root: str | Path = PROJECT_ROOT) -> list[ModelConfig]:
    return [load_model_config(path) for path in ladder_paths(project_root)]


def derive_rung(config: ModelConfig, vocab_size: int | None) -> ModelConfig:
    """``config`` with ``vocab_size`` and recomputed declared counts (no bound check).

    Mirrors :func:`epor.training.runner.derive_kernel_model`, which the sweep
    still calls for every rung it actually trains.
    """

    if vocab_size is None or vocab_size == config.vocab_size:
        return config
    derived = ModelConfig.model_validate(
        {**config.model_dump(mode="json"), "vocab_size": vocab_size}
    )
    count = structural_parameter_count(derived)
    return ModelConfig.model_validate(
        {
            **derived.model_dump(mode="json"),
            "total_parameters": count,
            "active_parameters": count,
            "core_parameters": count,
        }
    )


def rung_status(config: ModelConfig) -> RungStatus:
    """Whether the local kernel may allocate ``config``; never raises for size."""

    counts = parameter_counts(config)
    local = (
        config.architecture == "dense-decoder" and counts.total <= REFERENCE_MODEL_MAX_PARAMETERS
    )
    if local:
        reason = "within the local training kernel's dense 50,000,000-parameter bound"
    elif config.architecture != "dense-decoder":
        reason = "the local training kernel trains dense decoders only; projection only"
    else:
        reason = (
            f"{counts.total:,} structural parameters exceed the local kernel's "
            f"{REFERENCE_MODEL_MAX_PARAMETERS:,}-parameter bound; projection only"
        )
    return RungStatus(
        slug=config.slug,
        config_sha256=config.sha256,
        nominal_non_embedding=LADDER_RUNGS.get(config.slug),
        total_parameters=counts.total,
        non_embedding_parameters=counts.non_embedding,
        vocab_size=config.vocab_size,
        execution="local" if local else "projection-only",
        reason=reason,
    )


__all__ = [
    "LADDER_DIRECTORY",
    "LADDER_RUNGS",
    "LADDER_TOLERANCE",
    "ExecutionStatus",
    "RungStatus",
    "derive_rung",
    "ladder_paths",
    "load_ladder",
    "rung_status",
]
