"""Compute/cost projections: hand-checked arithmetic, unknown inputs, and no network."""

from __future__ import annotations

import math
import socket
from pathlib import Path
from typing import Any

import pytest

from epor.models.config import load_model_config
from epor.scaling.fit import Observation, fit_observations
from epor.scaling.projection import (
    DEFAULT_FAILURE_RESERVE_FRACTION,
    DEFAULT_TOKENS_PER_PARAMETER,
    ProjectionInputs,
    load_projection_inputs,
    project_training,
)

TINY = load_model_config("configs/models/epor-tiny.yaml")
EXAMPLE = Path("configs/scaling/projection-inputs.example.yaml")
GIB = 1024**3


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: Any, **__: Any) -> None:
        raise AssertionError("projection attempted network access")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def _value(section: dict[str, Any], key: str) -> float | None:
    return section[key]["value"]


def test_projection_arithmetic_is_hand_checkable() -> None:
    inputs = ProjectionInputs.model_validate(
        {
            "hardware": {
                "provider": "example-provider",
                "device_sku": "example-sku",
                "device_count": 2,
                "peak_flops_per_device": 1e12,
                "assumed_mfu": 0.5,
                "mfu_source": "test fixture",
                "device_memory_bytes": 1_000_000_000,
                "device_power_kw": 0.5,
            },
            "prices": {
                "currency": "TEST",
                "price_per_device_hour": 2.0,
                "storage_price_per_gib_month": 0.1,
                "egress_price_per_gib": 0.05,
            },
            "energy": {"pue": 1.2, "carbon_intensity_kg_per_kwh": 0.4},
            "run": {
                "sequence_length": 16,
                "token_budget": 1_000_000,
                "checkpoint_every_hours": 0.0001,
                "keep_checkpoints": 3,
                "evaluation_every_tokens": 100_000,
                "evaluation_tokens_per_pass": 1_000,
                "egress_gib": 10.0,
                "storage_months": 2.0,
                "failure_reserve_fraction": 0.1,
            },
        }
    )
    report = project_training(TINY, inputs)
    results = report.results
    assert results["unknown_inputs"] == []

    train_per_token = 3 * (2 * 114_944 + 2 * 2 * 16 * 64)  # 701,952
    forward_per_token = 2 * 114_944 + 2 * 2 * 16 * 64
    training_flops = train_per_token * 1_000_000
    effective = 1e12 * 0.5
    training_hours = training_flops / effective / 3600
    evaluation_hours = 10 * 1_000 * forward_per_token / effective / 3600
    total_hours = training_hours + evaluation_hours
    wall = total_hours / 2
    written = math.ceil(wall / 0.0001)
    checkpoint = 115_264 * 12
    retained = min(written, 3) * checkpoint
    shards = 2 * 1_000_000
    storage_cost = (retained + shards) / GIB * 2.0 * 0.1
    compute_cost = total_hours * 2.0
    subtotal = compute_cost + storage_cost + 10.0 * 0.05
    energy = total_hours * 0.5 * 1.2

    assert _value(results["training_flops"], "train_flops_per_token") == train_per_token
    assert _value(results["training_flops"], "total") == pytest.approx(training_flops)
    gpu = results["gpu_hours"]
    assert _value(gpu, "training_device_hours") == pytest.approx(training_hours)
    assert _value(gpu, "evaluation_device_hours") == pytest.approx(evaluation_hours)
    assert _value(gpu, "total_device_hours") == pytest.approx(total_hours)
    assert _value(gpu, "wall_clock_hours") == pytest.approx(wall)
    assert _value(gpu, "projected_tokens_per_second") == pytest.approx(
        2 * effective / train_per_token
    )
    assert _value(results["memory"], "training_state_bytes") == 115_264 * 16
    assert _value(results["storage"], "checkpoint_bytes") == checkpoint
    assert _value(results["checkpoint_cadence"], "written") == written
    assert _value(results["storage"], "retained_checkpoint_bytes") == retained
    assert _value(results["storage"], "token_shard_bytes") == shards
    assert _value(results["storage"], "cost") == pytest.approx(storage_cost)
    assert _value(results["egress"], "cost") == pytest.approx(0.5)
    assert _value(results["cost"], "compute") == pytest.approx(compute_cost)
    assert _value(results["cost"], "total_with_reserve") == pytest.approx(subtotal * 1.1)
    assert _value(results["failure_reserve"], "cost") == pytest.approx(subtotal * 0.1)
    assert _value(results["energy_carbon"], "energy_kwh") == pytest.approx(energy)
    assert _value(results["energy_carbon"], "carbon_kg") == pytest.approx(energy * 0.4)
    envelope = results["budget_abort_envelope"]
    assert envelope["max_spend"]["value"] == pytest.approx(subtotal * 1.1)
    assert envelope["max_device_hours"]["value"] == pytest.approx(total_hours * 1.1)
    assert len(envelope["mandatory_abort_conditions"]) == 8
    assert results["approval"] == {
        "required": True,
        "approval_id": None,
        "statement": results["approval"]["statement"],
    }
    assert report.scope == "projection"
    assert report.kind == "compute-projection"
    assert project_training(TINY, inputs).report_id == report.report_id


def test_projection_covers_the_compute_policy_proposal_list() -> None:
    results = project_training(TINY, ProjectionInputs()).results
    for key in (
        "tokens",
        "gpu_hours",
        "provider",
        "storage",
        "egress",
        "evaluation",
        "checkpoint_cadence",
        "failure_reserve",
        "energy_carbon",
        "budget_abort_envelope",
    ):
        assert key in results


def test_missing_operator_inputs_are_unknown_not_defaulted() -> None:
    report = project_training(TINY, load_projection_inputs(EXAMPLE))
    results = report.results
    # The shipped example is all nulls.
    assert report.inputs["operator_inputs"]["hardware"]["peak_flops_per_device"] is None
    tokens = results["tokens"]["training_tokens"]
    assert tokens["value"] == DEFAULT_TOKENS_PER_PARAMETER * 115_264
    assert tokens["label"] == "hypothesis"
    assert results["failure_reserve"]["fraction"]["value"] == DEFAULT_FAILURE_RESERVE_FRACTION
    for section, key in (
        ("training_flops", "total"),
        ("gpu_hours", "total_device_hours"),
        ("gpu_hours", "wall_clock_hours"),
        ("cost", "total_with_reserve"),
        ("energy_carbon", "carbon_kg"),
        ("storage", "cost"),
    ):
        estimate = results[section][key]
        assert estimate["value"] is None
        assert estimate["label"] == "unknown"
        assert estimate["missing"]
    assert "prices.price_per_device_hour" in results["unknown_inputs"]
    assert "hardware.peak_flops_per_device" in results["unknown_inputs"]
    assert results["provider"]["label"] == "unknown"
    assert results["budget_abort_envelope"]["expected_final_loss"]["value"] is None
    assert any("unknown" in warning for warning in report.warnings)


def test_projection_extrapolates_loss_from_a_fit_with_caveat() -> None:
    observations = [
        Observation(parameters=n, tokens=1e9, loss=2.0 + 50.0 * n**-0.3)
        for n in (1e5, 3e5, 1e6, 3e6)
    ]
    fit = fit_observations(observations, source={"kind": "synthetic"}, scope="harness-fixture")
    ladder_1b = load_model_config("configs/models/ladder/epor-ladder-1b.yaml")
    report = project_training(ladder_1b, ProjectionInputs(), fit_report=fit)
    loss = report.results["budget_abort_envelope"]["expected_final_loss"]
    assert loss["label"] == "hypothesis"
    assert loss["extrapolation_factor"] > 100
    assert report.inputs["fit_report_id"] == fit.report_id
    assert report.results["target"]["local_execution"] == "projection-only"


def test_inputs_file_is_strict(tmp_path: Path) -> None:
    bad = tmp_path / "inputs.yaml"
    bad.write_text("hardware:\n  peak_flops: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="peak_flops"):
        load_projection_inputs(bad)
