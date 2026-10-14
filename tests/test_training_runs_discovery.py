"""Bounded discovery of nested training runs and evaluation reports."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from conftest import PROJECT_ROOT, make_token_shards

from epor.control import training_runs
from epor.control.schemas import TrainingSummaryRead
from epor.control.training_runs import training_summary

POINT = "ab" * 32


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One verified two-step kernel run with an evaluation report, to copy around."""

    pytest.importorskip("torch")
    from epor.training import PackedTrainingSettings, evaluate_packed_checkpoint
    from epor.training.runner import derive_kernel_model, run_kernel

    root = tmp_path_factory.mktemp("trained")
    shards = make_token_shards(root / "shards")
    model = derive_kernel_model(
        PROJECT_ROOT / "configs/models/epor-tiny.yaml",
        data_root=shards.data_root,
        tokenized_id=shards.tokenized_id,
        vocab_from_build=True,
    )
    settings = PackedTrainingSettings(
        tokenized_id=shards.tokenized_id,
        seq_len=16,
        tokens_per_batch=32,
        precision="fp32",
        checkpoint_every=2,
        eval_max_steps=1,
    )
    run_dir = root / "run"
    run_kernel(model, shards.data_root, run_dir, settings, max_steps=2)
    evaluate_packed_checkpoint(run_dir, shards.data_root, report_dir=run_dir / "evaluations")
    return run_dir


def place(run: Path, destination: Path, **fields: Any) -> Path:
    """Copy a run; ``fields`` rewrite its live manifest and immutable copy alike."""

    shutil.copytree(run, destination)
    if fields:
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        immutable = destination / "manifests" / f"{manifest['run_id']}.json"
        for path in (destination / "manifest.json", immutable):
            path.write_text(json.dumps({**manifest, **fields}), encoding="utf-8")
    return destination


def synthetic(directory: Path, started_at: str = "2026-10-01T00:00:00Z", **fields: Any) -> Path:
    """A running kernel manifest with no checkpoint yet: valid without torch."""

    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "command": "kernel_train",
        "run_id": f"kernel-{directory.name}",
        "status": "running",
        "started_at": started_at,
        **fields,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def roots(tmp_path: Path) -> tuple[Path, Path]:
    artifacts = tmp_path / ".epor" / "artifacts"
    artifacts.mkdir(parents=True)
    (tmp_path / "runs").mkdir()
    return artifacts, tmp_path


def test_nested_runs_and_reports_are_listed_once_with_their_location(
    trained_run: Path, tmp_path: Path
) -> None:
    artifacts, project = roots(tmp_path)
    place(trained_run, artifacts / "job-1" / "kernel")
    for attempt in (1, 2):
        place(trained_run, artifacts / "job-2" / "sweep" / "runs" / POINT / f"attempt-{attempt}")
    place(trained_run, project / "runs" / "align" / "sft", chat_dataset_id="c" * 64)
    place(trained_run, project / "runs" / "align" / "dpo", command="alignment_dpo")
    nested = place(trained_run, project / "runs" / "a" / "b")
    # A manifest inside a checkpoint or immutable-copy folder is never a run.
    checkpoint = next((nested / "checkpoints").glob("step-*"))
    shutil.copyfile(nested / "manifest.json", checkpoint / "manifest.json")
    shutil.copyfile(nested / "manifest.json", nested / "manifests" / "manifest.json")

    summary = training_summary(artifacts, project)

    assert summary["integrity_errors"] == [] and summary["notices"] == []
    locations = [(run["source"], run["location"]) for run in summary["runs"]]
    assert sorted(locations) == sorted(
        [
            ("job", "job-1/kernel"),
            ("job", f"job-2/sweep/runs/{POINT}/attempt-1"),
            ("job", f"job-2/sweep/runs/{POINT}/attempt-2"),
            ("cli", "runs/align/sft"),
            ("cli", "runs/align/dpo"),
            ("cli", "runs/a/b"),
        ]
    )
    objectives = {run["location"]: run["objective"] for run in summary["runs"]}
    assert objectives["runs/align/sft"] == "sft" and objectives["runs/align/dpo"] == "dpo"
    assert all(run["checkpoint"]["step"] == 2 for run in summary["runs"])
    report_dirs = sorted(str(Path(item["location"]).parent) for item in summary["evaluations"])
    assert report_dirs == sorted(f"{location}/evaluations" for _, location in locations)
    TrainingSummaryRead.model_validate(summary)


def test_tampered_nested_run_is_an_integrity_error(trained_run: Path, tmp_path: Path) -> None:
    artifacts, project = roots(tmp_path)
    attempt = place(trained_run, project / "runs" / "sweep" / "runs" / POINT / "attempt-1")
    trainer = next((attempt / "checkpoints").glob("step-*")) / "trainer.json"
    trainer.write_bytes(trainer.read_bytes().replace(b'"step": 2', b'"step": 3'))

    summary = training_summary(artifacts, project)

    assert summary["runs"] == []
    [error] = summary["integrity_errors"]
    assert f"cli run runs/sweep/runs/{POINT}/attempt-1" in error and "digest" in error


def test_checkpoint_symlink_out_of_the_run_is_an_integrity_error(
    trained_run: Path, tmp_path: Path
) -> None:
    artifacts, project = roots(tmp_path)
    run = place(trained_run, project / "runs" / "a" / "b")
    outside = tmp_path / "outside-checkpoints"
    (run / "checkpoints").rename(outside)
    (run / "checkpoints").mkdir()
    (run / "checkpoints" / "latest.json").symlink_to(outside / "latest.json")

    [error] = training_summary(artifacts, project)["integrity_errors"]

    assert error.startswith("cli run runs/a/b:") and "beneath" in error


def test_symlinks_out_of_a_root_are_ignored(tmp_path: Path) -> None:
    artifacts, project = roots(tmp_path)
    outside = synthetic(tmp_path / "elsewhere" / "run")
    (outside / "evaluations").mkdir()
    (outside / "evaluations" / "report.json").write_text(
        json.dumps({"command": "kernel_eval", "created_at": "2026-10-01T00:00:00Z"}),
        encoding="utf-8",
    )
    (project / "runs" / "linked-run").symlink_to(outside, target_is_directory=True)
    (artifacts / "job-1" / "linked").mkdir(parents=True)
    (artifacts / "job-1" / "linked" / "manifest.json").symlink_to(outside / "manifest.json")
    (project / "runs" / "x" / "evaluations").mkdir(parents=True)
    (project / "runs" / "x" / "evaluations" / "report.json").symlink_to(
        outside / "evaluations" / "report.json"
    )

    summary = training_summary(artifacts, project)

    assert summary["runs"] == [] and summary["evaluations"] == []
    assert summary["integrity_errors"] == [] and summary["notices"] == []


def test_a_run_id_cannot_redirect_the_immutable_copy(tmp_path: Path) -> None:
    artifacts, project = roots(tmp_path)
    # "../manifest" would make the live manifest its own immutable copy.
    synthetic(project / "runs" / "a", status="succeeded", run_id="../manifest")

    summary = training_summary(artifacts, project)

    assert summary["runs"] == []
    assert summary["integrity_errors"] == ["cli run runs/a: run manifest has an invalid run_id"]


def test_depth_bound_is_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifacts, project = roots(tmp_path)
    synthetic(project / "runs" / "a" / "b")
    synthetic(project / "runs" / "a" / "b" / "c" / "d")
    monkeypatch.setattr(training_runs, "MAX_DEPTH", 2)

    summary = training_summary(artifacts, project)

    assert [run["location"] for run in summary["runs"]] == ["runs/a/b"]
    assert summary["notices"] == [
        "cli discovery: directories more than 2 levels deep were not searched; "
        "the listing may be incomplete"
    ]


def test_manifest_and_directory_counts_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts, project = roots(tmp_path)
    for name in ("a", "b", "c"):
        synthetic(project / "runs" / name)
    monkeypatch.setattr(training_runs, "MAX_MANIFESTS", 2)

    summary = training_summary(artifacts, project)

    assert sorted(run["location"] for run in summary["runs"]) == ["runs/a", "runs/b"]
    assert summary["notices"] == [
        "cli discovery: only the first 2 run manifests were read; the listing may be incomplete"
    ]

    monkeypatch.setattr(training_runs, "MAX_DIRECTORIES", 2)
    summary = training_summary(artifacts, project)
    assert [run["location"] for run in summary["runs"]] == ["runs/a"]
    assert summary["notices"] == [
        "cli discovery: the search stopped after 2 directories; the listing may be incomplete"
    ]


def test_max_runs_keeps_the_newest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifacts, project = roots(tmp_path)
    for day in (1, 2, 3):
        synthetic(project / "runs" / "sweep" / f"run-{day}", f"2026-10-0{day}T00:00:00Z")
    monkeypatch.setattr(training_runs, "MAX_RUNS", 2)

    summary = training_summary(artifacts, project)

    assert [run["location"] for run in summary["runs"]] == ["runs/sweep/run-3", "runs/sweep/run-2"]
    assert summary["notices"] == ["showing the newest 2 of 3 runs"]


def test_overlapping_roots_list_a_run_once(tmp_path: Path) -> None:
    (tmp_path / "runs").mkdir()
    artifacts = tmp_path / "runs" / "artifacts"
    synthetic(artifacts / "job-1" / "kernel")

    summary = training_summary(artifacts, tmp_path)

    assert [(run["source"], run["location"]) for run in summary["runs"]] == [
        ("job", "job-1/kernel")
    ]
