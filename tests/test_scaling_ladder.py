"""Scaling ladder parameter counts, rung execution status, and the analytic FLOPs model."""

from __future__ import annotations

from pathlib import Path

import pytest
from test_model_config_features import PINNED_CONFIG_SHA256

from epor.models.config import ModelConfig, load_model_config, structural_parameter_count
from epor.scaling.flops import estimate_flops, parameter_counts
from epor.scaling.ladder import (
    LADDER_RUNGS,
    LADDER_TOLERANCE,
    derive_rung,
    ladder_paths,
    load_ladder,
    rung_status,
)

CONFIG_ROOT = Path("configs/models")


def test_ladder_rungs_land_within_tolerance_of_their_nominal_size() -> None:
    ladder = load_ladder()
    assert [config.slug for config in ladder] == list(LADDER_RUNGS)
    measured = {}
    for config in ladder:
        counts = parameter_counts(config)
        nominal = LADDER_RUNGS[config.slug]
        assert abs(counts.non_embedding - nominal) / nominal <= LADDER_TOLERANCE
        # Declared totals are the structural count, and the ladder targets the 65,536 vocab.
        assert config.total_parameters == structural_parameter_count(config) == counts.total
        assert config.vocab_size == 65_536
        assert config.architecture == "dense-decoder"
        assert config.trained_max_context == config.validated_max_context == 0
        measured[config.slug] = counts.non_embedding
    assert measured == {
        "epor-ladder-10m": 10_326_912,
        "epor-ladder-100m": 103_243_008,
        "epor-ladder-300m": 301_517_824,
        "epor-ladder-1b": 1_013_012_480,
    }


def test_shipped_config_hashes_are_unchanged_by_the_ladder() -> None:
    for filename, expected in PINNED_CONFIG_SHA256.items():
        assert load_model_config(CONFIG_ROOT / filename).sha256 == expected
    ladder_hashes = {config.sha256 for config in load_ladder()}
    assert not ladder_hashes & set(PINNED_CONFIG_SHA256.values())
    assert all(path.parent.name == "ladder" for path in ladder_paths())


def test_only_rungs_within_the_kernel_bound_run_locally() -> None:
    statuses = {config.slug: rung_status(config) for config in load_ladder()}
    assert statuses["epor-ladder-10m"].execution == "local"
    for slug in ("epor-ladder-100m", "epor-ladder-300m", "epor-ladder-1b"):
        assert statuses[slug].execution == "projection-only"
        assert "projection only" in statuses[slug].reason
    # A proxy vocabulary does not move a projection-only rung under the bound.
    small = derive_rung(load_ladder()[1], 1_024)
    assert rung_status(small).execution == "projection-only"
    assert small.total_parameters == structural_parameter_count(small)
    alpha = load_model_config(CONFIG_ROOT / "epor-alpha.yaml")
    assert rung_status(alpha).execution == "projection-only"


def test_flops_model_matches_a_hand_count_for_the_tiny_config() -> None:
    tiny = load_model_config(CONFIG_ROOT / "epor-tiny.yaml")
    # d=64, L=2, heads=4 (head 16), kv=2 (width 32), d_ff=192, V=260.
    attention = 2 * 64 * 64 + 2 * 64 * 32  # 12,288
    feed_forward = 3 * 64 * 192  # 36,864
    matmul = 2 * (attention + feed_forward) + 64 * 260  # 114,944
    assert matmul == 114_944
    estimate = estimate_flops(tiny, 16)
    assert estimate.matmul_parameters == matmul
    assert estimate.attention_flops_per_token == 2 * 2 * 16 * 64
    assert estimate.forward_flops_per_token == 2 * matmul + 4_096 == 233_984
    assert estimate.train_flops_per_token == 3 * 233_984
    assert estimate.mtp_forward_flops_per_token == 0
    counts = parameter_counts(tiny)
    assert counts.total == 115_264
    assert counts.non_embedding == 115_264 - 260 * 64
    assert counts.active == counts.total


def test_flops_model_counts_mtp_and_active_experts() -> None:
    tiny = load_model_config(CONFIG_ROOT / "epor-tiny.yaml")
    mtp = ModelConfig.model_validate({**tiny.model_dump(mode="json"), "mtp_depth": 1})
    base = estimate_flops(tiny, 16)
    extra = estimate_flops(mtp, 16)
    per_depth_matmul = 2 * 64 * 64 + 49_152 + 64 * 260
    assert extra.matmul_parameters - base.matmul_parameters == per_depth_matmul
    assert extra.mtp_forward_flops_per_token == 2 * per_depth_matmul + 2 * 16 * 64
    assert extra.forward_flops_per_token - base.forward_flops_per_token == (
        extra.mtp_forward_flops_per_token
    )

    alpha = load_model_config(CONFIG_ROOT / "epor-alpha.yaml")
    counts = parameter_counts(alpha)
    expert = 3 * alpha.d_model * alpha.d_ff
    assert counts.total - counts.active == alpha.n_layers * expert * (
        alpha.n_routed_experts - alpha.experts_per_token
    )
    with pytest.raises(ValueError, match="positive"):
        estimate_flops(tiny, 0)
