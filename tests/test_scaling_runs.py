"""Hardware profile, resumable sweeps, and data-mixture ablations on tiny fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from conftest import TokenShardFixture, make_domain_shards, make_token_shards  # noqa: E402
from kernel_support import kernel_config, kernel_settings  # noqa: E402

from epor.data.store import DataStore  # noqa: E402
from epor.scaling import (  # noqa: E402
    MixturePlan,
    NamedMixture,
    ProfileSettings,
    ScalingReportError,
    SweepPlan,
    fit_sweep,
    load_scaling_reports,
    mixture_summary,
    profile_training_step,
    run_mixture_ablation,
    run_sweep,
    store_scaling_report,
)
from epor.scaling.report import report_bytes  # noqa: E402
from epor.training import TrainingCancelled  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TINY = "configs/models/epor-tiny.yaml"
LADDER_100M = "configs/models/ladder/epor-ladder-100m.yaml"


@pytest.fixture(scope="module")
def shards(tmp_path_factory: pytest.TempPathFactory) -> TokenShardFixture:
    return make_token_shards(tmp_path_factory.mktemp("scaling"))


# --------------------------------------------------------------------------- profile


def test_profile_records_throughput_and_unknown_mfu_without_a_peak() -> None:
    config = kernel_config(1_024)
    settings = ProfileSettings(batch_size=2, seq_len=16, warmup_steps=1, timed_steps=2)
    report = profile_training_step(config, settings)
    assert report.kind == "hardware-profile"
    assert report.scope == "host-measurement"
    measured = report.measured
    assert measured is not None
    assert measured["tokens_per_second"] > 0
    assert len(measured["step_seconds"]) == 2
    assert measured["mfu"] is None
    assert measured["mfu_status"] == "unknown"
    assert measured["process_peak_rss_bytes"] > 0
    assert measured["host"]["device"] == "cpu"
    assert report.results["peak_flops_label"] == "unknown"
    assert report.inputs["determinism"]["torch_deterministic_algorithms"] is True
    assert (
        measured["achieved_flops_per_second"]
        == measured["tokens_per_second"] * report.results["flops"]["train_flops_per_token"]
    )
    # Timings are excluded from the content address.
    again = profile_training_step(config, settings)
    assert again.report_id == report.report_id
    assert again.measured != report.measured

    with_peak = profile_training_step(
        config, settings.model_copy(update={"peak_flops_per_second": 1e12})
    )
    assert with_peak.measured is not None
    assert with_peak.measured["mfu_status"] == "measured"
    assert with_peak.measured["mfu"] == pytest.approx(
        with_peak.measured["achieved_flops_per_second"] / 1e12
    )
    assert with_peak.report_id != report.report_id


def test_profile_refuses_oversized_models_and_honours_cancellation() -> None:
    from epor.models.config import load_model_config

    with pytest.raises(ValueError, match="50,000,000"):
        profile_training_step(
            load_model_config(PROJECT_ROOT / LADDER_100M),
            ProfileSettings(batch_size=1, seq_len=8, warmup_steps=0, timed_steps=1),
        )
    with pytest.raises(TrainingCancelled):
        profile_training_step(
            kernel_config(1_024), ProfileSettings(seq_len=8), should_cancel=lambda: True
        )


def test_reports_store_immutably_and_detect_tampering(tmp_path: Path) -> None:
    report = profile_training_step(
        kernel_config(1_024), ProfileSettings(batch_size=1, seq_len=8, timed_steps=1)
    )
    store = DataStore(tmp_path / "data")
    path = store_scaling_report(store, report)
    assert path == (
        tmp_path / "data" / "scaling-reports" / report.report_id / f"{report.content_sha256}.json"
    )
    assert store_scaling_report(store, report) == path
    assert load_scaling_reports(store, report.report_id) == [report]
    payload = json.loads(path.read_bytes())
    payload["results"]["timed_steps"] = 99
    path.chmod(0o600)
    path.write_bytes(json.dumps(payload).encode())
    with pytest.raises(ScalingReportError):
        load_scaling_reports(store, report.report_id)
    assert report_bytes(report)


# --------------------------------------------------------------------------- sweep


def _plan(shards: TokenShardFixture) -> SweepPlan:
    training = kernel_settings(shards.tokenized_id, validation_split="validation", eval_max_steps=1)
    return SweepPlan(
        rungs=(TINY, LADDER_100M), token_budgets=(64,), seeds=(0, 1), training=training
    )


def _attempts(output: Path) -> list[str]:
    return sorted(p.relative_to(output).as_posix() for p in output.glob("runs/*/attempt-*"))


def test_sweep_runs_local_points_marks_projection_only_and_resumes(
    shards: TokenShardFixture, tmp_path: Path
) -> None:
    plan = _plan(shards)
    seen: list[tuple[int, int, str]] = []
    output = tmp_path / "sweep"
    report = run_sweep(
        plan,
        data_root=shards.data_root,
        output_dir=output,
        project_root=PROJECT_ROOT,
        on_point=lambda done, total, point: seen.append((done, total, point["status"])),
        scope="harness-fixture",
    )
    points = report.results["points"]
    assert [p["status"] for p in points] == [
        "completed",
        "completed",
        "projection-only",
        "projection-only",
    ]
    assert report.results["counts"] == {"completed": 2, "projection-only": 2, "failed": 0}
    assert [s[:2] for s in seen] == [(1, 4), (2, 4), (3, 4), (4, 4)]
    local = points[0]
    assert local["max_steps"] == 2 and local["tokens_seen"] == 64
    assert local["vocab_size"] == shards.vocab_size
    assert local["validation_main_loss"] is not None
    assert [step for step, _ in local["loss_curve"]] == [1.0, 2.0]
    assert points[0]["point_id"] != points[1]["point_id"]
    assert points[0]["final_main_loss"] != points[1]["final_main_loss"]
    projected = points[2]
    assert projected["parameters_total"] > 50_000_000
    assert "final_main_loss" not in projected
    assert "projection only" in projected["reason"]
    assert any("projection-only" in warning for warning in report.warnings)
    attempts = _attempts(output)
    assert len(attempts) == 2

    # A rerun skips completed points and reproduces the same report ID.
    rerun = run_sweep(
        plan,
        data_root=shards.data_root,
        output_dir=output,
        project_root=PROJECT_ROOT,
        should_cancel=lambda: False,
        scope="harness-fixture",
    )
    assert rerun.report_id == report.report_id
    assert _attempts(output) == attempts

    fit = fit_sweep(report)
    assert fit.inputs["source"]["report_id"] == report.report_id
    assert fit.inputs["source"]["loss"] == "validation_main_loss"
    assert fit.results["loss_vs_parameters"]["status"] == "insufficient-points"


def test_cancelled_sweep_resumes_to_the_uninterrupted_result(
    shards: TokenShardFixture, tmp_path: Path
) -> None:
    plan = _plan(shards).model_copy(update={"rungs": (TINY,), "seeds": (0,)})
    baseline = run_sweep(
        plan,
        data_root=shards.data_root,
        output_dir=tmp_path / "baseline",
        project_root=PROJECT_ROOT,
        scope="harness-fixture",
    )
    calls = {"count": 0}

    def cancel_after_first_step() -> bool:
        calls["count"] += 1
        return calls["count"] == 3  # sweep check, before step 1, after step 1

    output = tmp_path / "interrupted"
    with pytest.raises(TrainingCancelled):
        run_sweep(
            plan,
            data_root=shards.data_root,
            output_dir=output,
            project_root=PROJECT_ROOT,
            should_cancel=cancel_after_first_step,
            scope="harness-fixture",
        )
    assert not list(output.glob("points/*.json"))
    resumed = run_sweep(
        plan,
        data_root=shards.data_root,
        output_dir=output,
        project_root=PROJECT_ROOT,
        scope="harness-fixture",
    )
    assert _attempts(output) == [f"runs/{resumed.results['points'][0]['point_id']}/attempt-1"]
    assert resumed.report_id == baseline.report_id
    assert resumed.results["points"] == baseline.results["points"]


# --------------------------------------------------------------------------- mixtures


@pytest.fixture(scope="module")
def mixed_build(tmp_path_factory: pytest.TempPathFactory) -> TokenShardFixture:
    return make_domain_shards(tmp_path_factory.mktemp("mixture"))


def test_mixture_ablation_reports_per_domain_validation_loss(
    mixed_build: TokenShardFixture, tmp_path: Path
) -> None:
    plan = MixturePlan(
        model=TINY,
        mixtures=(
            NamedMixture(mixture_id="balanced", weights={"code": 1.0, "general": 1.0}),
            NamedMixture(mixture_id="code-only", weights={"code": 1.0, "general": 0.0}),
        ),
        max_steps=2,
        evaluation_max_steps=2,
        training=kernel_settings(mixed_build.tokenized_id),
    )
    output = tmp_path / "mixtures"
    report = run_mixture_ablation(
        plan,
        data_root=mixed_build.data_root,
        output_dir=output,
        project_root=PROJECT_ROOT,
        scope="harness-fixture",
    )
    assert report.kind == "mixture-ablation"
    lineage: dict[str, Any] = report.inputs["lineage"]
    assert lineage["tokenized_id"] == mixed_build.tokenized_id
    summary = mixture_summary(report)
    assert [item["mixture_id"] for item in summary] == ["balanced", "code-only"]
    for item in summary:
        assert set(item["by_domain"]) == {"code", "general"}
        assert all(value > 0 for value in item["by_domain"].values())
    balanced, code_only = report.results["mixtures"]
    assert balanced["selection"]["selection_sha256"] != code_only["selection"]["selection_sha256"]
    assert balanced["tokens_seen"] == code_only["tokens_seen"] == 64
    assert code_only["epochs_over_subset"] > balanced["epochs_over_subset"]
    # The training run recorded the mixture it trained on.
    manifest = json.loads(
        next(output.glob(f"runs/{code_only['point_id']}/attempt-1/manifest.json")).read_text()
    )
    assert manifest["train_mixture"]["mixture_sha256"] == code_only["mixture_sha256"]

    rerun = run_mixture_ablation(
        plan,
        data_root=mixed_build.data_root,
        output_dir=output,
        project_root=PROJECT_ROOT,
        scope="harness-fixture",
    )
    assert rerun.report_id == report.report_id
