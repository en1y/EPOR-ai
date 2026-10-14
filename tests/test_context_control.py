"""Context-laboratory control jobs, their scopes and results, and the /context endpoint."""

from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Thread
from typing import Any

import pytest
from conftest import Harness, run
from test_laboratory_control import TINY, _run, _service

from epor.control.authority import ANONYMOUS, Actor, Role
from epor.control.errors import (
    AuthenticationRequiredError,
    AuthorizationDeniedError,
    InvalidJobSpecError,
)
from epor.control.models import JobStatus, JobType

CONTEXT_JOBS = (
    JobType.CONTEXT_EXTEND,
    JobType.CONTEXT_EVALUATE,
    JobType.CONTEXT_PROFILE,
    JobType.CONTEXT_SWEEP,
)
H = "0" * 64


def _clean(result: dict[str, Any] | None) -> dict[str, Any]:
    assert result is not None
    assert "[REDACTED]" not in json.dumps(result)
    return result


class Lab:
    def __init__(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        from conftest import make_token_shards

        from epor.control.worker import JobWorker

        self.settings, self.service, self.owner = _service(tmp_path)
        self.shards = make_token_shards(tmp_path, max_shard_tokens=4096)
        self.worker = JobWorker(self.settings, service=self.service, worker_id="context-worker")

    def job(self, job_type: JobType, spec: dict[str, Any]) -> dict[str, Any]:
        job = self.service.create_job(job_type, spec, actor=self.owner)
        done = _run(self.worker, job)
        assert done.status is JobStatus.SUCCEEDED, done.error_message
        return _clean(done.result)


@pytest.fixture(scope="module")
def lab(tmp_path_factory: pytest.TempPathFactory) -> Lab:
    state = Lab(tmp_path_factory.mktemp("context-control"))
    base = state.job(
        JobType.KERNEL_TRAIN,
        {
            "tokenized_id": state.shards.tokenized_id,
            "model_config": TINY,
            "vocab_from_build": True,
            "max_steps": 2,
            "seq_len": 64,
            "tokens_per_batch": 128,
            "precision": "fp32",
            "checkpoint_every": 2,
            "validation_split": None,
        },
    )
    state.base = base["output_path"]  # type: ignore[attr-defined]
    return state


def test_every_context_job_has_a_handler_and_certification_is_not_a_job(tmp_path: Path) -> None:
    from epor.control.worker import DEFAULT_HANDLERS

    assert set(CONTEXT_JOBS) <= set(DEFAULT_HANDLERS)
    _settings, service, owner = _service(tmp_path)
    for action_id in ("context_certify", "context_project"):
        with pytest.raises(InvalidJobSpecError, match="not a control-plane job type"):
            service.create_job(action_id, {}, actor=owner)


def test_context_jobs_need_their_own_scope(tmp_path: Path) -> None:
    _settings, service, _owner = _service(tmp_path)
    (tmp_path / ".epor" / "artifacts" / "run").mkdir(parents=True)
    specs: dict[JobType, dict[str, Any]] = {
        JobType.CONTEXT_EXTEND: {"mode": "extend", "checkpoint_path": "run", "target_context": 256},
        JobType.CONTEXT_EVALUATE: {"checkpoint_path": "run", "lengths": [256]},
        JobType.CONTEXT_PROFILE: {"mode": "feasibility"},
        JobType.CONTEXT_SWEEP: {"axis": "window", "values": [16], "tokenized_id": H},
    }
    trainer = Actor(id="t", role=Role.DELEGATED_OPERATOR, scopes=frozenset({"kernel_train"}))
    evaluator = Actor(id="e", role=Role.DELEGATED_OPERATOR, scopes=frozenset({"eval_suite"}))
    for job_type, spec in specs.items():
        for outsider in (trainer, evaluator):
            with pytest.raises(AuthorizationDeniedError):
                service.create_job(job_type, spec, actor=outsider)
        with pytest.raises(AuthenticationRequiredError):
            service.create_job(job_type, spec, actor=ANONYMOUS)
        scoped = Actor(id="s", role=Role.DELEGATED_OPERATOR, scopes=frozenset({job_type.value}))
        assert service.create_job(job_type, spec, actor=scoped).status is JobStatus.QUEUED


def test_context_job_specs_are_typed_bounded_and_contained(tmp_path: Path) -> None:
    _settings, service, owner = _service(tmp_path)
    (tmp_path / ".epor" / "artifacts" / "run").mkdir(parents=True)
    for job_type, bad in (
        (JobType.CONTEXT_EXTEND, {"mode": "extend", "checkpoint_path": "run"}),
        (
            JobType.CONTEXT_EXTEND,
            {"mode": "extend", "checkpoint_path": "../x", "target_context": 256},
        ),
        (
            JobType.CONTEXT_EXTEND,
            {"mode": "extend", "checkpoint_path": "run", "target_context": 256, "strategy": "ddp"},
        ),
        (JobType.CONTEXT_EVALUATE, {"checkpoint_path": "run", "lengths": [32]}),
        (JobType.CONTEXT_EVALUATE, {"checkpoint_path": "missing", "lengths": [256]}),
        (JobType.CONTEXT_PROFILE, {"mode": "profile", "lengths": [256]}),
        (JobType.CONTEXT_PROFILE, {"mode": "feasibility", "model_config": "../x.yaml"}),
        (JobType.CONTEXT_PROFILE, {"mode": "feasibility", "min_length": 4096, "max_length": 1024}),
        (JobType.CONTEXT_SWEEP, {"axis": "depth", "values": [1], "tokenized_id": H}),
        (JobType.CONTEXT_SWEEP, {"axis": "window", "values": [], "tokenized_id": H}),
    ):
        with pytest.raises(InvalidJobSpecError):
            service.create_job(job_type, bad, actor=owner)
    assert service.list_jobs() == []


def test_the_context_pipeline_runs_as_jobs(lab: Lab) -> None:
    base = lab.base  # type: ignore[attr-defined]
    extended = lab.job(
        JobType.CONTEXT_EXTEND,
        {
            "mode": "extend",
            "checkpoint_path": base,
            "target_context": 384,
            "max_steps": 2,
            "precision": "fp32",
            "tokens_per_batch": 768,
        },
    )
    assert (extended["configured_context"], extended["trained_context"]) == (384, 384)
    assert extended["validated_context_raised"] is False
    evaluated = lab.job(
        JobType.CONTEXT_EVALUATE,
        {
            "checkpoint_path": extended["output_path"],
            "lengths": [256],
            "items_per_cell": 1,
            "split": "train",
        },
    )
    assert set(evaluated["accuracy"]) >= {"passkey", "prompt-injection"}
    profiled = lab.job(
        JobType.CONTEXT_PROFILE,
        {
            "mode": "profile",
            "checkpoint_path": extended["output_path"],
            "lengths": [256],
            "decode_tokens": 4,
        },
    )
    assert profiled["statuses"] == {"256": "ok"}
    feasible = lab.job(
        JobType.CONTEXT_PROFILE,
        {
            "mode": "feasibility",
            "variants": ["full", "local"],
            "min_length": 128,
            "max_length": 256,
            "window": 64,
        },
    )
    assert feasible["feasible_maximum"]["full"]["inference"] == 256
    swept = lab.job(
        JobType.CONTEXT_SWEEP,
        {
            "axis": "window",
            "values": [32],
            "tokenized_id": lab.shards.tokenized_id,
            "vocab_from_build": True,
            "seq_len": 64,
            "tokens_per_batch": 128,
            "max_steps": 1,
            "eval_length": 256,
            "items_per_cell": 1,
        },
    )
    assert swept["points"] == 1
    from epor.context.summary import context_summary

    assert lab.settings.data_root is not None and lab.settings.artifact_root is not None
    summary = context_summary(
        lab.settings.data_root, lab.settings.artifact_root, lab.settings.project_root
    )
    rows = {row["location"]: row for row in summary["runs"]}
    assert (
        rows[extended["output_path"]]["trained"] == 384
        and rows[extended["output_path"]]["validated"] == 0
    )
    assert summary["g6"] and summary["profile"] and summary["feasibility"] and summary["sweep"]
    assert summary["integrity_errors"] == []


def test_extension_job_cancellation_is_cooperative(lab: Lab) -> None:
    job = lab.service.create_job(
        JobType.CONTEXT_EXTEND,
        {
            "mode": "extend",
            "checkpoint_path": lab.base,  # type: ignore[attr-defined]
            "target_context": 256,
            "max_steps": 5_000,
            "precision": "fp32",
            "checkpoint_every": 10_000,
        },
        actor=lab.owner,
    )
    thread = Thread(target=lab.worker.run_once)
    thread.start()
    deadline = time.monotonic() + 60
    while (
        time.monotonic() < deadline and lab.service.get_job(job.id).status is not JobStatus.RUNNING
    ):
        time.sleep(0.02)
    time.sleep(0.5)
    lab.service.cancel(job.id)
    thread.join(timeout=120)
    assert not thread.is_alive()
    assert lab.service.get_job(job.id).status is JobStatus.CANCELLED


def test_the_context_endpoint_is_read_only_and_verified(harness: Harness) -> None:
    captured: dict[str, Any] = {}

    async def fetch() -> None:
        async with harness.client() as client:
            captured["get"] = await client.get("/api/v1/context")
            captured["post"] = await client.post("/api/v1/context")

    run(fetch)
    assert captured["get"].status_code == 200
    body = captured["get"].json()
    assert "only a G6 certificate raises validated context" in body["scope"]
    assert body["runs"] == [] and body["integrity_errors"] == []
    assert captured["post"].status_code == 405
