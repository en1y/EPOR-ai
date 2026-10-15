"""Projections for the progressive 32K → 128K → 256K context recipe.

Nothing above the locally feasible length is trained on the reference
machine.  For each stage of :data:`~epor.context.extension.PROGRESSIVE_RECIPE`
this module reports, from a target config and operator-written inputs:

* the KV-cache bytes at that length for the target as configured, for an
  interleaved local/global variant, and for int8 storage of each — analytic,
  from :func:`epor.models.cache.kv_cache_bytes` (``EPOR-adaptation``);
* attention FLOPs per token, with local layers attending to at most their
  window (Kaplan's ``2 n_layer n_ctx d`` attention term per layer,
  ``disclosed``, with ``n_ctx`` bounded per layer as an ``EPOR-adaptation``);
* the full v0.0.6 training projection at that sequence length (tokens, FLOPs,
  device-hours, memory, storage, energy, cost), where every quantity whose
  operator input is missing stays ``unknown``.

A projection is not a measured, trained, or validated result, and not an
approval: a target run still needs a named human's explicit approval of the
exact proposal (G5), and 256K is for α/β only after 128K passes G6.
"""

from __future__ import annotations

from typing import Any

from epor.models.cache import kv_cache_bytes
from epor.models.config import ModelConfig
from epor.scaling.projection import ProjectionInputs, project_training
from epor.scaling.report import KAPLAN_2020, ScalingReport, seal_scaling_report

from .citations import DEEPSEEK_V2, GEMMA3, YARN
from .extension import PROGRESSIVE_RECIPE

# Gemma 3's disclosed local configuration, used only to show what an
# interleaved variant of the target would cache.
REFERENCE_LOCAL = {"sliding_window": 1024, "local_layers_per_global": 5}


def attention_flops_per_token(config: ModelConfig, length: int) -> int:
    total = 0
    for layer in range(config.n_layers):
        reach = min(length, config.sliding_window) if config.layer_is_local(layer) else length
        total += 2 * reach * config.d_model
    return total


def local_variant(config: ModelConfig) -> ModelConfig:
    if config.local_layers_per_global:
        return config
    ratio = min(REFERENCE_LOCAL["local_layers_per_global"], max(config.n_layers - 1, 1))
    return ModelConfig.model_validate(
        {
            **config.model_dump(mode="json"),
            "sliding_window": min(REFERENCE_LOCAL["sliding_window"], config.configured_max_context),
            "local_layers_per_global": ratio,
        }
    )


def project_context(
    target: ModelConfig,
    inputs: ProjectionInputs,
    *,
    stages: tuple[int, ...] = PROGRESSIVE_RECIPE[1:],
    feasibility: dict[str, Any] | None = None,
) -> ScalingReport:
    variants = {"as-configured": target, "interleaved-local-global": local_variant(target)}
    rows: list[dict[str, Any]] = []
    warnings = [
        "Projection only: no stage above the locally feasible length was trained or "
        "validated, and validated context rises only with a passing G6 report.",
    ]
    for length in stages:
        staged = ModelConfig.model_validate(
            {
                **target.model_dump(mode="json"),
                "configured_max_context": max(length, target.configured_max_context),
            }
        )
        training = project_training(
            staged,
            inputs.model_copy(
                update={"run": inputs.run.model_copy(update={"sequence_length": length})}
            ),
        )
        cache = {
            name: {
                "bf16_bytes": kv_cache_bytes(config, length, dtype_bytes=2),
                "int8_bytes": kv_cache_bytes(config, length, quantize="int8"),
                "bf16_bytes_per_token": kv_cache_bytes(config, length, dtype_bytes=2) / length,
                "attention_flops_per_token": attention_flops_per_token(config, length),
                "label": "EPOR-adaptation",
            }
            for name, config in variants.items()
        }
        rows.append(
            {
                "length": length,
                "stage_rule": (
                    "alpha/beta only, after 128K passes G6"
                    if length > 131_072
                    else "progressive stage"
                ),
                "kv_cache": cache,
                "training": training.results,
                "training_warnings": training.warnings,
            }
        )
    return seal_scaling_report(
        kind="context-projection",
        scope="projection",
        inputs={
            "target_config_sha256": target.sha256,
            "target_model_id": target.slug,
            "stages": list(stages),
            "projection_inputs": inputs.model_dump(mode="json"),
            "local_variant": REFERENCE_LOCAL,
            "locally_feasible": feasibility,
        },
        results={"stages": rows},
        warnings=warnings,
        citations=[KAPLAN_2020, GEMMA3, YARN, DEEPSEEK_V2],
    )


__all__ = ["REFERENCE_LOCAL", "attention_flops_per_token", "local_variant", "project_context"]
