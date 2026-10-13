"""Training-kernel jobs through covenant admission, the worker, and the read API."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from threading import Thread
from typing import Any

import pytest

pytest.importorskip("torch")

from conftest import PROJECT_ROOT, Harness, TokenShardFixture, make_token_shards, run

from epor.control.authority import ANONYMOUS, Actor, AuthorityStore, Role
from epor.control.database import (
    create_session_factory,
    create_sqlite_engine,
    initialize_database,
)
from epor.control.errors import (
    AuthenticationRequiredError,
    AuthorizationDeniedError,
    InvalidJobSpecError,
)
from epor.control.models import Job, JobStatus, JobType
from epor.control.security import redact
from epor.control.service import JobService
from epor.control.settings import ControlSettings
from epor.control.worker import JobWorker

TINY = "configs/models/epor-tiny.yaml"


def kernel_service(
    tmp_path: Path,
) -> tuple[ControlSettings, JobService, Actor, TokenShardFixture]:
    settings = ControlSettings(
        project_root=tmp_path,
        database_path=tmp_path / ".epor/control.sqlite3",
        artifact_root=tmp_path / ".epor/artifacts",
        safety_root=tmp_path / ".epor/control",
        data_root=tmp_path / "data",
        research_catalog_path=tmp_path / "research/catalog.yaml",
        poll_interval_seconds=0.005,
    )
    settings.prepare_directories()
    assert settings.database_path is not None
    assert settings.safety_root is not None
    (tmp_path / "configs" / "models").mkdir(parents=True)
    shutil.copyfile(PROJECT_ROOT / TINY, tmp_path / TINY)
    owner = AuthorityStore(settings.safety_root).bootstrap_owner().principal.actor()
    engine = create_sqlite_engine(settings.database_path)
    initialize_database(engine)
    service = JobService(create_session_factory(engine), settings)
    shards = make_token_shards(tmp_path, max_shard_tokens=4096)
    return settings, service, owner, shards


def train_spec(tokenized_id: str, **overrides: Any) -> dict[str, Any]:
    return {
        "tokenized_id": tokenized_id,
        "model_config": TINY,
        "vocab_from_build": True,
        "max_steps": 4,
        "total_steps": 8,
        "seq_len": 16,
        "tokens_per_batch": 32,
        "micro_batch_size": 1,
        "precision": "fp32",
        "learning_rate": 3e-3,
        "checkpoint_every": 2,
        "eval_max_steps": 1,
        **overrides,
    }


def run_job(worker: JobWorker, expected: Job) -> Job:
    completed = worker.run_once()
    assert completed is not None
    assert completed.id == expected.id
    return completed


def test_kernel_jobs_train_resume_and_evaluate_through_the_worker(tmp_path: Path) -> None:
    settings, service, owner, shards = kernel_service(tmp_path)
    assert settings.artifact_root is not None
    worker = JobWorker(settings, service=service, worker_id="kernel-worker")

    train = service.create_job(
        JobType.KERNEL_TRAIN, train_spec(shards.tokenized_id, max_steps=2), actor=owner
    )
    trained = run_job(worker, train)
    assert trained.status is JobStatus.SUCCEEDED, trained.error_message
    result = trained.result
    assert result is not None
    assert result["steps_completed"] == 2 and result["start_step"] == 0
    # Token measurements survive redaction; they are not credentials.
    assert result["tokens_seen"] == 2 * 32
    assert result["tokens_per_batch"] == 32
    assert result["evaluation"]["target_tokens"] > 0
    assert result["model"]["vocab_size"] == shards.vocab_size
    assert result["strategy"] == "none" and "proxy-scale" in result["scope"]
    kinds = {item.kind for item in service.list_artifacts(train.id)}
    assert {"run-manifest", "training-metrics", "checkpoint-manifest"} <= kinds
    progress = [
        event.payload
        for event in service.list_events(train.id)
        if event.kind == "job.progress" and "step" in event.payload
    ]
    assert [item["step"] for item in progress] == [1, 2]

    # Resuming from another job's run continues the lineage in a new output.
    resume = service.create_job(
        JobType.KERNEL_TRAIN,
        train_spec(shards.tokenized_id, resume_from=result["output_path"]),
        actor=owner,
    )
    resumed = run_job(worker, resume)
    assert resumed.status is JobStatus.SUCCEEDED, resumed.error_message
    assert resumed.result is not None
    assert resumed.result["start_step"] == 2 and resumed.result["steps_completed"] == 4
    assert resumed.result["lineage_id"] == result["lineage_id"]

    # A resume that would change the trajectory is refused by the trainer.
    drift = service.create_job(
        JobType.KERNEL_TRAIN,
        train_spec(shards.tokenized_id, resume_from=result["output_path"], learning_rate=1e-3),
        actor=owner,
    )
    refused = run_job(worker, drift)
    assert refused.status is JobStatus.FAILED
    assert "trajectory" in (refused.error_message or "")

    evaluate = service.create_job(
        JobType.KERNEL_EVALUATE,
        {"checkpoint_path": resumed.result["output_path"], "split": "validation"},
        actor=owner,
    )
    evaluated = run_job(worker, evaluate)
    assert evaluated.status is JobStatus.SUCCEEDED, evaluated.error_message
    assert evaluated.result is not None
    assert evaluated.result["split"] == "validation"
    assert evaluated.result["target_tokens"] > 0
    assert evaluated.result["checkpoint_path"].endswith("step-00000004")
    [report] = service.list_artifacts(evaluate.id)
    assert report.kind == "evaluation-report"


def test_vocabulary_mismatch_fails_without_touching_the_recipe(tmp_path: Path) -> None:
    settings, service, owner, shards = kernel_service(tmp_path)
    recipe = (tmp_path / TINY).read_bytes()
    worker = JobWorker(settings, service=service, worker_id="kernel-worker")
    job = service.create_job(
        JobType.KERNEL_TRAIN,
        train_spec(shards.tokenized_id, vocab_from_build=False),
        actor=owner,
    )
    failed = run_job(worker, job)
    assert failed.status is JobStatus.FAILED
    assert "vocab_from_build" in (failed.error_message or "")
    assert (tmp_path / TINY).read_bytes() == recipe


def test_resume_paths_stay_beneath_the_artifact_root(tmp_path: Path) -> None:
    settings, service, owner, shards = kernel_service(tmp_path)
    worker = JobWorker(settings, service=service, worker_id="kernel-worker")
    for escape in ("../../", "/etc"):
        job = service.create_job(
            JobType.KERNEL_TRAIN,
            train_spec(shards.tokenized_id, resume_from=escape),
            actor=owner,
        )
        assert run_job(worker, job).status is JobStatus.FAILED
        evaluate = service.create_job(
            JobType.KERNEL_EVALUATE, {"checkpoint_path": escape}, actor=owner
        )
        assert run_job(worker, evaluate).status is JobStatus.FAILED


def test_kernel_job_cancels_after_checkpointing_its_last_step(tmp_path: Path) -> None:
    settings, service, owner, shards = kernel_service(tmp_path)
    assert settings.artifact_root is not None
    worker = JobWorker(settings, service=service, worker_id="kernel-worker")
    job = service.create_job(
        JobType.KERNEL_TRAIN,
        train_spec(shards.tokenized_id, max_steps=2_000, total_steps=2_000, checkpoint_every=1_000),
        actor=owner,
    )
    thread = Thread(target=worker.run_once)
    thread.start()
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        steps = [
            event.payload.get("step")
            for event in service.list_events(job.id)
            if event.kind == "job.progress"
        ]
        if any(isinstance(step, int) and step >= 2 for step in steps):
            break
        time.sleep(0.02)
    service.cancel(job.id)
    thread.join(timeout=60)
    assert not thread.is_alive()
    assert service.get_job(job.id).status is JobStatus.CANCELLED

    run_dir = settings.artifact_root / job.id / "kernel"
    pointer = json.loads((run_dir / "checkpoints" / "latest.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "cancelled"
    assert 2 <= pointer["step"] < 2_000
    assert manifest["steps_completed"] == pointer["step"]


@pytest.mark.parametrize(
    ("job_type", "spec"),
    [
        (JobType.KERNEL_TRAIN, {"tokenized_id": "0" * 64}),
        (JobType.KERNEL_EVALUATE, {"checkpoint_path": "job/kernel"}),
    ],
)
def test_kernel_jobs_refuse_unauthorized_actors_before_parsing(
    tmp_path: Path, job_type: JobType, spec: dict[str, Any]
) -> None:
    _settings, service, _owner, _shards = kernel_service(tmp_path)
    sharder = Actor(
        id="sharder", role=Role.DELEGATED_OPERATOR, scopes=frozenset({"tokenizer_shard"})
    )

    with pytest.raises(AuthorizationDeniedError):
        service.create_job(job_type, spec, actor=sharder)
    with pytest.raises(AuthenticationRequiredError):
        service.create_job(job_type, spec, actor=ANONYMOUS)
    assert service.list_jobs() == []

    scoped = Actor(id="k", role=Role.DELEGATED_OPERATOR, scopes=frozenset({job_type.value}))
    assert service.create_job(job_type, spec, actor=scoped).status is JobStatus.QUEUED


def test_kernel_job_specs_are_typed_and_bounded(tmp_path: Path) -> None:
    _settings, service, owner, shards = kernel_service(tmp_path)
    base = train_spec(shards.tokenized_id)
    for bad in (
        {**base, "tokenized_id": "../escape"},
        {**base, "strategy": "ddp"},
        {**base, "tokens_per_batch": 2_000_000},
        {**base, "tokens_per_batch": 40},
        {**base, "max_steps": 100_001},
        {**base, "warmup_steps": 9},
        {**base, "model_config": "configs/models/epor-alpha.yaml"},
        {**base, "fault_plan": {"step": 1}},
        {**base, "command": "rm -rf /"},
    ):
        with pytest.raises(InvalidJobSpecError):
            service.create_job(JobType.KERNEL_TRAIN, bad, actor=owner)
    with pytest.raises(InvalidJobSpecError):
        service.create_job(
            JobType.KERNEL_EVALUATE, {"checkpoint_path": "x", "split": "all"}, actor=owner
        )
    assert service.list_jobs() == []


def test_training_token_measurements_are_allowlisted_by_exact_key() -> None:
    assert redact(
        {
            "tokens_seen": 4096,
            "tokens_per_batch": 32,
            "target_tokens": 31,
            "tokens_seen_secret": "value",
            "target_token": "value",
            "tokens_per_batch_token": "value",
        }
    ) == {
        "tokens_seen": 4096,
        "tokens_per_batch": 32,
        "target_tokens": 31,
        "tokens_seen_secret": "[REDACTED]",
        "target_token": "[REDACTED]",
        "tokens_per_batch_token": "[REDACTED]",
    }


def test_training_summary_lists_verified_runs_and_flags_tampering(harness: Harness) -> None:
    from epor.training import PackedTrainingSettings, evaluate_packed_checkpoint
    from epor.training.runner import derive_kernel_model, run_kernel

    assert harness.settings.artifact_root is not None
    shards = make_token_shards(harness.settings.artifact_root.parent / "shards")
    model = derive_kernel_model(
        PROJECT_ROOT / TINY,
        data_root=shards.data_root,
        tokenized_id=shards.tokenized_id,
        vocab_from_build=True,
        mtp_depth=1,
    )
    settings = PackedTrainingSettings(
        tokenized_id=shards.tokenized_id,
        seq_len=16,
        tokens_per_batch=32,
        precision="fp32",
        checkpoint_every=2,
        eval_max_steps=1,
    )
    run_dir = harness.settings.artifact_root / "job-1" / "kernel"
    run_kernel(model, shards.data_root, run_dir, settings, max_steps=2)
    evaluate_packed_checkpoint(
        run_dir,
        shards.data_root,
        report_dir=harness.settings.artifact_root / "job-2" / "kernel-eval",
    )

    summaries: list[dict[str, Any]] = []

    async def fetch() -> None:
        async with harness.client() as client:
            summaries.append((await client.get("/api/v1/training/runs")).json())

    run(fetch)
    summary = summaries[-1]
    assert summary["integrity_errors"] == []
    assert "proxy-scale" in summary["scope"].lower()
    [listed] = [item for item in summary["runs"] if item["source"] == "job"]
    assert listed["location"] == "job-1/kernel"
    assert listed["status"] == "succeeded"
    assert listed["checkpoint"]["step"] == 2
    assert listed["mtp_depth"] == 1 and listed["qk_norm"] is False
    assert listed["tokens_seen"] == 64
    assert len(listed["mtp_losses"]) == 1
    [report] = [item for item in summary["evaluations"] if item["location"].startswith("job-2")]
    assert report["checkpoint_step"] == 2 and len(report["mtp_losses"]) == 1

    trainer = run_dir / "checkpoints" / "step-00000002" / "trainer.json"
    trainer.write_bytes(trainer.read_bytes().replace(b'"step": 2', b'"step": 3'))
    run(fetch)
    summary = summaries[-1]
    assert not [item for item in summary["runs"] if item["source"] == "job"]
    assert any("job-1/kernel" in error for error in summary["integrity_errors"])
