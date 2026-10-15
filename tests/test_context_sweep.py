"""Context sweeps against a full-attention control, and the 32K/128K/256K projection."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from conftest import make_token_shards  # noqa: E402
from kernel_support import kernel_config  # noqa: E402

from epor.context.projection import (  # noqa: E402
    attention_flops_per_token,
    local_variant,
    project_context,
)
from epor.context.sweep import ContextSweepPlan, run_context_sweep, variant  # noqa: E402
from epor.models.config import load_model_config  # noqa: E402
from epor.scaling.projection import ProjectionInputs  # noqa: E402


def _plan(axis: str, values: tuple[float, ...], tokenized_id: str = "0" * 64) -> ContextSweepPlan:
    return ContextSweepPlan(
        axis=axis,  # type: ignore[arg-type]
        values=values,
        window=16,
        ratio=1,
        tokenized_id=tokenized_id,
        seq_len=64,
        tokens_per_batch=128,
        max_steps=2,
        eval_length=256,
        items_per_cell=1,
    )


def test_each_axis_varies_one_thing_against_the_control() -> None:
    base = kernel_config(1024, n_layers=4)
    control = variant(base, _plan("window", (8,)), None)
    assert (control.sliding_window, control.local_layers_per_global) == (0, 0)
    assert control.configured_max_context == 320
    assert variant(base, _plan("window", (8,)), 8).sliding_window == 8
    assert variant(base, _plan("ratio", (3,)), 3).local_layers_per_global == 3
    split = variant(base, _plan("rope-base", (5000.0, 0)), 5000.0)
    assert split.local_rope_theta == 5000.0 and split.sliding_window == 16
    assert variant(base, _plan("rope-base", (0,)), 0).local_rope_theta is None
    with pytest.raises(ValueError, match="must not repeat"):
        _plan("window", (8, 8))


def test_a_window_sweep_trains_the_control_and_reports_kv_bytes_and_retrieval(
    tmp_path: Path,
) -> None:
    shards = make_token_shards(tmp_path, max_shard_tokens=4096)
    plan = _plan("window", (16,), shards.tokenized_id)
    base = kernel_config(shards.vocab_size)
    report = run_context_sweep(
        base, plan, data_root=shards.data_root, output_dir=tmp_path / "sweep"
    )
    assert report.kind == "context-sweep" and report.scope == "proxy-scale"
    control, [point] = report.results["control"], report.results["points"]
    assert control["name"] == "control-full-attention" and point["sliding_window"] == 16
    assert point["kv_bytes_per_token_bf16"] < control["kv_bytes_per_token_bf16"]
    for row in (control, point):
        assert set(row["retrieval"]) == {"passkey", "niah-multikey"}
        assert set(row["retrieval"]["passkey"]) == {"beginning", "middle", "end"}
    again = run_context_sweep(base, plan, data_root=shards.data_root, output_dir=tmp_path / "sweep")
    assert again.report_id == report.report_id  # finished points are reused, results identical
    # An impossible point (no global layer left in two layers) is refused
    # before the control or any other point trains.
    impossible = _plan("ratio", (1, 3), shards.tokenized_id)
    with pytest.raises(ValueError, match="no global attention layer"):
        run_context_sweep(base, impossible, data_root=shards.data_root, output_dir=tmp_path / "x")
    assert not (tmp_path / "x").exists()


def test_projection_reports_kv_bytes_per_stage_and_keeps_unknowns_unknown() -> None:
    beta = load_model_config("configs/models/epor-beta.yaml")
    report = project_context(beta, ProjectionInputs())
    assert report.kind == "context-projection" and report.scope == "projection"
    stages = report.results["stages"]
    assert [row["length"] for row in stages] == [32_768, 131_072, 262_144]
    full = [row["kv_cache"]["as-configured"]["bf16_bytes"] for row in stages]
    assert full == sorted(full) and full[0] < full[-1]
    for row in stages:
        local = row["kv_cache"]["interleaved-local-global"]
        assert local["bf16_bytes"] < row["kv_cache"]["as-configured"]["bf16_bytes"]
        assert local["int8_bytes"] < local["bf16_bytes"]
        assert row["training"]["gpu_hours"]["total_device_hours"]["label"] == "unknown"
        assert (
            "run.sequence_length"
            not in row["training"]["gpu_hours"]["total_device_hours"]["missing"]
        )
    assert stages[-1]["stage_rule"].startswith("alpha/beta only")
    windowed = local_variant(beta)
    assert attention_flops_per_token(windowed, 262_144) < attention_flops_per_token(beta, 262_144)
