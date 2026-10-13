"""Evaluation and scaling-laboratory jobs, their redaction, and the read-only report API."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from threading import Thread
from typing import Any

import pytest
from conftest import PROJECT_ROOT, Harness, TokenShardFixture, run
from evaluation_support import TEST_HARNESS, small_suite, small_suite_model

from epor.control.authority import ANONYMOUS, Actor, AuthorityStore, Role
from epor.control.database import create_session_factory, create_sqlite_engine, initialize_database
from epor.control.errors import (
    AuthenticationRequiredError,
    AuthorizationDeniedError,
    InvalidJobSpecError,
)
from epor.control.models import Job, JobStatus, JobType
from epor.control.security import redact, redact_values
from epor.control.service import JobService
from epor.control.settings import ControlSettings
from epor.control.worker import JobWorker

TINY = "configs/models/epor-tiny.yaml"
LADDER_100M = "configs/models/ladder/epor-ladder-100m.yaml"
LABORATORY_JOBS = (
    JobType.EVAL_SUITE,
    JobType.EVAL_CONTAMINATION,
    JobType.SCALE_PROFILE,
    JobType.SCALE_SWEEP,
    JobType.SCALE_MIXTURE,
)


def _service(tmp_path: Path) -> tuple[ControlSettings, JobService, Actor]:
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
    assert settings.database_path is not None and settings.safety_root is not None
    (tmp_path / "configs" / "models" / "ladder").mkdir(parents=True)
    shutil.copyfile(PROJECT_ROOT / TINY, tmp_path / TINY)
    shutil.copyfile(PROJECT_ROOT / LADDER_100M, tmp_path / LADDER_100M)
    owner = AuthorityStore(settings.safety_root).bootstrap_owner().principal.actor()
    engine = create_sqlite_engine(settings.database_path)
    initialize_database(engine)
    return settings, JobService(create_session_factory(engine), settings), owner


def _run(worker: JobWorker, expected: Job) -> Job:
    completed = worker.run_once()
    assert completed is not None and completed.id == expected.id
    return completed


def _template(tokenized_id: str, **overrides: Any) -> dict[str, Any]:
    return {
        "tokenized_id": tokenized_id,
        "seq_len": 16,
        "tokens_per_batch": 32,
        "micro_batch_size": 1,
        "precision": "fp32",
        "learning_rate": 3e-3,
        "checkpoint_every": 2,
        "validation_split": None,
        **overrides,
    }


def _no_redaction(result: dict[str, Any] | None) -> None:
    assert result is not None
    assert "[REDACTED]" not in json.dumps(result)


@pytest.fixture
def kernel_lab(
    tmp_path: Path,
) -> tuple[ControlSettings, JobService, Actor, TokenShardFixture, JobWorker, str]:
    pytest.importorskip("torch")
    from conftest import make_token_shards

    settings, service, owner = _service(tmp_path)
    shards = make_token_shards(tmp_path, max_shard_tokens=4096)
    worker = JobWorker(settings, service=service, worker_id="lab-worker")
    train = service.create_job(
        JobType.KERNEL_TRAIN,
        {
            "tokenized_id": shards.tokenized_id,
            "model_config": TINY,
            "vocab_from_build": True,
            "max_steps": 2,
            "seq_len": 16,
            "tokens_per_batch": 32,
            "precision": "fp32",
            "checkpoint_every": 2,
            "validation_split": "validation",
            "eval_max_steps": 1,
        },
        actor=owner,
    )
    trained = _run(worker, train)
    assert trained.status is JobStatus.SUCCEEDED, trained.error_message
    assert trained.result is not None
    return settings, service, owner, shards, worker, trained.result["output_path"]


def test_evaluation_jobs_store_reports_and_never_execute_code(kernel_lab: Any) -> None:
    settings, service, owner, shards, worker, checkpoint = kernel_lab
    job = service.create_job(
        JobType.EVAL_SUITE, {"checkpoint_path": checkpoint, "suite": "fixtures"}, actor=owner
    )
    done = _run(worker, job)
    assert done.status is JobStatus.SUCCEEDED, done.error_message
    result = done.result
    _no_redaction(result)
    assert result is not None
    assert result["scope"] == "harness-fixture"
    assert result["code_executed"] is False
    assert {"task_id": "fixture-code", "reason": "code execution not enabled"} in result[
        "skipped_tasks"
    ]
    assert (settings.data_root / "evaluation-reports" / f"{result['report_id']}.json").is_file()
    events = service.list_events(job.id)
    assert any(
        event.kind == "job.log" and "owner's CLI" in event.payload["message"] for event in events
    )
    tasks = [event.payload["task"] for event in events if "task" in event.payload]
    assert "fixture-multiple-choice" in tasks and "fixture-code" in tasks

    contamination = service.create_job(
        JobType.EVAL_CONTAMINATION, {"training_data": shards.tokenized_id}, actor=owner
    )
    scanned = _run(worker, contamination)
    assert scanned.status is JobStatus.SUCCEEDED, scanned.error_message
    _no_redaction(scanned.result)
    assert scanned.result is not None and scanned.result["model_kind"] == "data-only"


def test_code_execution_is_not_a_job(tmp_path: Path) -> None:
    _settings, service, owner = _service(tmp_path)
    # Declared, but only the owner's CLI may run it: the job API turns it away.
    with pytest.raises(InvalidJobSpecError, match="not a control-plane job type"):
        service.create_job("eval_code_execution", {}, actor=owner)
    for bad in (
        {"checkpoint_path": "run", "allow_code_execution": True},
        {"checkpoint_path": "run", "allow_unisolated_network": True},
        {"checkpoint_path": "../escape"},
        {"checkpoint_path": "run", "suite": "/etc/passwd"},
        {"checkpoint_path": "run", "suite": "configs/evaluation/../../x.yaml"},
        {"checkpoint_path": "run", "sources": [{"source_id": "b", "path": "../outside.jsonl"}]},
        {"checkpoint_path": "run", "training_data": "not-a-hash"},
        {"checkpoint_path": "run", "batch_size": 0},
    ):
        with pytest.raises(InvalidJobSpecError):
            service.create_job(JobType.EVAL_SUITE, bad, actor=owner)
    assert service.list_jobs() == []


def test_laboratory_jobs_need_their_own_scope(tmp_path: Path) -> None:
    _settings, service, _owner = _service(tmp_path)
    specs: dict[JobType, dict[str, Any]] = {
        JobType.EVAL_SUITE: {"checkpoint_path": "run"},
        JobType.EVAL_CONTAMINATION: {"training_data": "0" * 64},
        JobType.SCALE_PROFILE: {},
        JobType.SCALE_SWEEP: {"token_budgets": [32], "training": _template("0" * 64)},
        JobType.SCALE_MIXTURE: {
            "mixtures": [{"mixture_id": "a", "weights": {"code": 1.0}}],
            "max_steps": 1,
            "training": _template("0" * 64),
        },
    }
    trainer = Actor(id="t", role=Role.DELEGATED_OPERATOR, scopes=frozenset({"kernel_train"}))
    for job_type, spec in specs.items():
        with pytest.raises(AuthorizationDeniedError):
            service.create_job(job_type, spec, actor=trainer)
        with pytest.raises(AuthenticationRequiredError):
            service.create_job(job_type, spec, actor=ANONYMOUS)
        scoped = Actor(id="s", role=Role.DELEGATED_OPERATOR, scopes=frozenset({job_type.value}))
        assert service.create_job(job_type, spec, actor=scoped).status is JobStatus.QUEUED


def test_scaling_job_specs_are_typed_and_bounded(tmp_path: Path) -> None:
    _settings, service, owner = _service(tmp_path)
    template = _template("0" * 64)
    sweep = {"token_budgets": [32], "training": template}
    for bad in (
        {**sweep, "token_budgets": list(range(1, 18))},
        {**sweep, "token_budgets": [32, 32]},
        {**sweep, "rungs": ["../configs/models/epor-tiny.yaml"]},
        {**sweep, "rungs": ["/etc/passwd"]},
        {**sweep, "token_budgets": [1, 2, 3, 4, 5, 6, 7, 8, 9], "seeds": list(range(8))},
        {**sweep, "training": {**template, "tokens_per_batch": 40}},
        {**sweep, "training": {**template, "train_mixture": {"weights": {"code": 1.0}}}},
        {**sweep, "output_name": "../x"},
    ):
        with pytest.raises(InvalidJobSpecError):
            service.create_job(JobType.SCALE_SWEEP, bad, actor=owner)
    mixture = {
        "mixtures": [{"mixture_id": "a", "weights": {"code": 1.0}}],
        "max_steps": 2,
        "training": template,
    }
    for bad in (
        {**mixture, "mixtures": []},
        {**mixture, "mixtures": [mixture["mixtures"][0]] * 2},
        {**mixture, "training": {**template, "warmup_steps": 3}},
        {**mixture, "model_config": "configs/other/x.yaml"},
    ):
        with pytest.raises(InvalidJobSpecError):
            service.create_job(JobType.SCALE_MIXTURE, bad, actor=owner)
    for bad in (
        {"timed_steps": 901},
        {"peak_flops_per_second": -1},
        {"model_config": "configs/models/../../x.yaml"},
    ):
        with pytest.raises(InvalidJobSpecError):
            service.create_job(JobType.SCALE_PROFILE, bad, actor=owner)
    # A kernel training job never trains a domain-weighted subset implicitly.
    with pytest.raises(InvalidJobSpecError):
        service.create_job(
            JobType.KERNEL_TRAIN,
            {"tokenized_id": "0" * 64, "train_mixture": {"weights": {"code": 1.0}}},
            actor=owner,
        )
    assert service.list_jobs() == []


def test_profile_and_sweep_jobs_store_scaling_reports(kernel_lab: Any) -> None:
    settings, service, owner, shards, worker, _checkpoint = kernel_lab
    profile = service.create_job(
        JobType.SCALE_PROFILE,
        {
            "model_config": TINY,
            "vocab_from_build": shards.tokenized_id,
            "batch_size": 1,
            "seq_len": 16,
            "warmup_steps": 0,
            "timed_steps": 1,
        },
        actor=owner,
    )
    profiled = _run(worker, profile)
    assert profiled.status is JobStatus.SUCCEEDED, profiled.error_message
    _no_redaction(profiled.result)
    assert profiled.result is not None
    assert profiled.result["tokens_per_second"] > 0
    assert profiled.result["mfu_status"] == "unknown"
    assert profiled.result["scope"] == "host-measurement"

    sweep = service.create_job(
        JobType.SCALE_SWEEP,
        {
            "rungs": [TINY, LADDER_100M],
            "token_budgets": [32],
            "training": _template(shards.tokenized_id),
        },
        actor=owner,
    )
    swept = _run(worker, sweep)
    assert swept.status is JobStatus.SUCCEEDED, swept.error_message
    _no_redaction(swept.result)
    result = swept.result
    assert result is not None
    assert result["completed_points"] == 1 and result["projection_only_points"] == 1
    assert result["output_path"] == f"{sweep.id}/sweep"
    report_dir = settings.data_root / "scaling-reports" / result["report_id"]
    assert (report_dir / f"{result['content_sha256']}.json").is_file()
    points = [event.payload for event in service.list_events(sweep.id) if "point" in event.payload]
    assert [item["status"] for item in points] == ["completed", "projection-only"]


def test_sweep_job_cancellation_is_cooperative(kernel_lab: Any) -> None:
    _settings, service, owner, shards, worker, _checkpoint = kernel_lab
    job = service.create_job(
        JobType.SCALE_SWEEP,
        {
            "rungs": [TINY],
            "token_budgets": [32 * 5_000],
            "training": _template(shards.tokenized_id, checkpoint_every=10_000),
        },
        actor=owner,
    )
    thread = Thread(target=worker.run_once)
    thread.start()
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and service.get_job(job.id).status is not JobStatus.RUNNING:
        time.sleep(0.02)
    time.sleep(0.5)
    service.cancel(job.id)
    thread.join(timeout=120)
    assert not thread.is_alive()
    assert service.get_job(job.id).status is JobStatus.CANCELLED


def test_mixture_job_trains_one_model_per_mixture(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from conftest import make_domain_shards

    settings, service, owner = _service(tmp_path)
    build = make_domain_shards(tmp_path)
    worker = JobWorker(settings, service=service, worker_id="lab-worker")
    job = service.create_job(
        JobType.SCALE_MIXTURE,
        {
            "model_config": TINY,
            "mixtures": [
                {"mixture_id": "balanced", "weights": {"code": 1.0, "general": 1.0}},
                {"mixture_id": "code-only", "weights": {"code": 1.0, "general": 0.0}},
            ],
            "max_steps": 2,
            "evaluation_max_steps": 1,
            "training": _template(build.tokenized_id),
        },
        actor=owner,
    )
    done = _run(worker, job)
    assert done.status is JobStatus.SUCCEEDED, done.error_message
    _no_redaction(done.result)
    assert done.result is not None and done.result["mixtures"] == 2


def test_laboratory_summary_keys_are_allowlisted_by_exact_key() -> None:
    assert redact(
        {
            "tokens_per_second": 1234.5,
            "token_per_second": "value",
            "tokens_per_second_token": "value",
            "secret_tokens_per_second": "value",
        }
    ) == {
        "tokens_per_second": 1234.5,
        "token_per_second": "[REDACTED]",
        "tokens_per_second_token": "[REDACTED]",
        "secret_tokens_per_second": "[REDACTED]",
    }
    # An allowlisted key does not exempt its value from the inline scan.
    leaked = redact({"tokens_per_second": "api_key=abcd1234efgh5678"})
    assert leaked == {"tokens_per_second": "api_key=[REDACTED]"}
    # The read-only report views scan values but keep schema keys legible.
    assert redact_values({"tokenizer_kind": "bearer abcdefgh12345678", "tokens": 3}) == {
        "tokenizer_kind": "bearer [REDACTED]",
        "tokens": 3,
    }


def _store_reports(data_root: Path) -> tuple[str, str, str]:
    from epor.data.store import DataStore
    from epor.evaluation import run_suite, store_report
    from epor.scaling import ProjectionInputs, load_ladder, project_training, store_scaling_report

    store = DataStore(data_root)
    first = run_suite(small_suite(), small_suite_model(), harness=TEST_HARNESS)
    second = run_suite(
        small_suite(), small_suite_model(second_correct=False, name="other"), harness=TEST_HARNESS
    )
    store_report(store, first)
    store_report(store, second)
    projection = project_training(load_ladder()[-1], ProjectionInputs())
    store_scaling_report(store, projection)
    return first.report_id, second.report_id, projection.report_id


def test_report_endpoints_verify_content_addresses(harness: Harness) -> None:
    assert harness.settings.data_root is not None
    data_root = harness.settings.data_root
    first, second, projection = _store_reports(data_root)
    captured: dict[str, Any] = {}

    async def fetch(*paths: str) -> None:
        async with harness.client() as client:
            for path in paths:
                response = await client.get(f"/api/v1{path}")
                captured[path] = (response.status_code, response.json())

    run(
        lambda: fetch(
            "/evaluation/reports",
            f"/evaluation/reports/{first}",
            f"/evaluation/compare?baseline={first}&candidate={second}&resamples=50",
            f"/evaluation/reports/{'0' * 64}",
            "/evaluation/reports/not-a-hash",
            "/scaling/reports",
            f"/scaling/reports/{projection}",
            "/scaling/ladder",
        )
    )
    status, listing = captured["/evaluation/reports"]
    assert status == 200 and listing["integrity_errors"] == []
    assert {item["report_id"] for item in listing["reports"]} == {first, second}
    status, view = captured[f"/evaluation/reports/{first}"]
    assert status == 200 and view["report_id"] == first
    assert all("items" not in task and task["item_count"] == 2 for task in view["tasks"])
    status, comparison = captured[
        f"/evaluation/compare?baseline={first}&candidate={second}&resamples=50"
    ]
    assert status == 200 and comparison["baseline_report_id"] == first
    assert captured[f"/evaluation/reports/{'0' * 64}"][0] == 404
    assert captured["/evaluation/reports/not-a-hash"][0] == 422
    status, scaling = captured["/scaling/reports"]
    assert status == 200 and scaling["integrity_errors"] == []
    [listed] = scaling["reports"]
    assert listed["kind"] == "compute-projection" and listed["approval_required"] is True
    status, [shown] = captured[f"/scaling/reports/{projection}"]
    assert status == 200 and shown["results"]["approval"]["required"] is True
    status, ladder = captured["/scaling/ladder"]
    assert status == 200
    assert {row["slug"]: row["execution"] for row in ladder["rungs"]}["epor-ladder-1b"] == (
        "projection-only"
    )

    # Tampering is reported, never served as healthy.
    stored = data_root / "evaluation-reports" / f"{first}.json"
    stored.write_bytes(stored.read_bytes().replace(b'"test-suite"', b'"edited-suite"', 1))
    [scaling_file] = (data_root / "scaling-reports" / projection).glob("*.json")
    scaling_file.write_bytes(scaling_file.read_bytes().replace(b"true", b"false", 1))
    captured.clear()
    run(lambda: fetch("/evaluation/reports", f"/evaluation/reports/{first}", "/scaling/reports"))
    status, listing = captured["/evaluation/reports"]
    assert {item["report_id"] for item in listing["reports"]} == {second}
    assert any(first in error for error in listing["integrity_errors"])
    assert captured[f"/evaluation/reports/{first}"][0] == 409
    status, scaling = captured["/scaling/reports"]
    assert scaling["reports"] == [] and len(scaling["integrity_errors"]) == 1


def test_every_laboratory_job_type_has_a_handler() -> None:
    from epor.control.worker import DEFAULT_HANDLERS

    assert set(LABORATORY_JOBS) <= set(DEFAULT_HANDLERS)
    assert "eval_code_execution" not in {job_type.value for job_type in JobType}
