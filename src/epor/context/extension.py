"""Derived configs for progressive context extension and the sparse-attention retrofit.

A shipped recipe is never edited.  Extension and retrofit runs train a
*derived* config whose hash differs from its parent's only in the fields the
kernel allows to change across an ``init_checkpoint``
(:data:`epor.training.kernel.EXTENSION_FIELDS`):

* :func:`extension_config` raises ``configured_max_context`` to the target and
  sets a RoPE scaling method whose factor is ``target / original`` — the
  length the parent lineage actually trained at, never its configured context;
* :func:`retrofit_config` adds a sparse-attention indexer, with declared
  parameter counts recomputed from the structure.

``PROGRESSIVE_RECIPE`` is the roadmap's stage plan (4/8K → 32K → 128K → 256K,
the last for α/β only after 128K passes G6).  On the CPU reference machine only
the first stages that fit are trained; the rest are projections
(:mod:`epor.context.projection`).
"""

from __future__ import annotations

from typing import Literal

from epor.models.config import (
    ModelConfig,
    RopeScaling,
    SparseAttention,
    structural_parameter_count,
)

ExtensionMethod = Literal["linear", "ntk", "yarn", "none"]
PROGRESSIVE_RECIPE: tuple[int, ...] = (8_192, 32_768, 131_072, 262_144)


def _with(parent: ModelConfig, **changes: object) -> ModelConfig:
    derived = ModelConfig.model_validate({**parent.model_dump(mode="json"), **changes})
    count = structural_parameter_count(derived)
    return ModelConfig.model_validate(
        {
            **derived.model_dump(mode="json"),
            "total_parameters": count,
            "active_parameters": count,
            "core_parameters": count,
        }
    )


def extension_config(
    parent: ModelConfig,
    target_context: int,
    *,
    trained_context: int,
    method: ExtensionMethod = "yarn",
) -> ModelConfig:
    """The config an extension run from ``parent`` to ``target_context`` trains."""

    if target_context <= trained_context:
        raise ValueError("an extension target must exceed the parent's trained context")
    factor = target_context / trained_context
    scaling: RopeScaling | None = None
    if method != "none":
        scaling = RopeScaling(type=method, factor=factor, original_max_context=trained_context)
    return _with(
        parent,
        configured_max_context=max(parent.configured_max_context, target_context),
        rope_scaling=None if scaling is None else scaling.model_dump(mode="json"),
    )


def retrofit_config(parent: ModelConfig, sparse: SparseAttention) -> ModelConfig:
    """``parent`` plus a sparse-attention indexer (GQA parents only)."""

    if parent.attention_variant != "gqa":
        raise ValueError("the sparse-attention retrofit applies to GQA checkpoints")
    return _with(parent, sparse_attention=sparse.model_dump(mode="json"))


__all__ = [
    "PROGRESSIVE_RECIPE",
    "ExtensionMethod",
    "extension_config",
    "retrofit_config",
]
