"""Alignment control jobs, their scopes and redaction, and the audit/report API."""

from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Thread
from typing import Any

import httpx
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

ALIGNMENT_JOBS = (
    JobType.ALIGN_GENERATE,
    JobType.ALIGN_JUDGE,
    JobType.ALIGN_DATASET,
    JobType.ALIGN_SFT,
    JobType.ALIGN_PREFERENCE,
    JobType.ALIGN_EVALUATE,
)
H = "0" * 64


def _clean(result: dict[str, Any] | None) -> dict[str, Any]:
    assert result is not None
    encoded = json.dumps(result)
    assert "[REDACTED]" not in encoded
    assert "output_token_ids" not in encoded  # model text and IDs never reach a job result
    return result


class Lab:
    def __init__(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        from conftest import make_token_shards

        from epor.control.worker import JobWorker

        self.settings, self.service, self.owner = _service(tmp_path)
        self.shards = make_token_shards(tmp_path, max_shard_tokens=4096)
        self.worker = JobWorker(self.settings, service=self.service, worker_id="align-worker")

    def job(self, job_type: JobType, spec: dict[str, Any]) -> dict[str, Any]:
        job = self.service.create_job(job_type, spec, actor=self.owner)
        done = _run(self.worker, job)
        assert done.status is JobStatus.SUCCEEDED, done.error_message
        return _clean(done.result)


@pytest.fixture(scope="module")
def lab(tmp_path_factory: pytest.TempPathFactory) -> Lab:
    from epor.alignment.fixtures import seed_fixtures
    from epor.alignment.records import AlignmentStore

    state = Lab(tmp_path_factory.mktemp("align-control"))
    base = state.job(
        JobType.KERNEL_TRAIN,
        {
            "tokenized_id": state.shards.tokenized_id,
            "model_config": TINY,
            "vocab_from_build": True,
            "max_steps": 2,
            "seq_len": 32,
            "tokens_per_batch": 64,
            "precision": "fp32",
            "checkpoint_every": 2,
            "validation_split": None,
        },
    )
    state.base = base["output_path"]  # type: ignore[attr-defined]
    assert state.settings.data_root is not None
    seed_fixtures(AlignmentStore(state.settings.data_root))  # the owner's `epor align fixtures`
    return state


def test_every_alignment_job_type_has_a_handler_and_two_actions_are_not_jobs(
    tmp_path: Path,
) -> None:
    from epor.control.worker import DEFAULT_HANDLERS

    assert set(ALIGNMENT_JOBS) <= set(DEFAULT_HANDLERS)
    _settings, service, owner = _service(tmp_path)
    for action_id in ("align_register", "align_audit"):
        with pytest.raises(InvalidJobSpecError, match="not a control-plane job type"):
            service.create_job(action_id, {}, actor=owner)
    assert service.list_jobs() == []


def test_alignment_jobs_need_their_own_scope(tmp_path: Path) -> None:
    _settings, service, _owner = _service(tmp_path)
    specs: dict[JobType, dict[str, Any]] = {
        JobType.ALIGN_GENERATE: {"mode": "eval-items", "checkpoint_path": "."},
        JobType.ALIGN_JUDGE: {"subject": "eval-item", "checkpoint_path": "."},
        JobType.ALIGN_DATASET: {"kind": "sft", "tokenizer_id": H, "include": ["fixtures"]},
        JobType.ALIGN_SFT: {"init_path": ".", "dataset_id": H},
        JobType.ALIGN_PREFERENCE: {"init_path": ".", "dataset_id": H},
        JobType.ALIGN_EVALUATE: {"mode": "calibration", "judging_run": H},
    }
    trainer = Actor(id="t", role=Role.DELEGATED_OPERATOR, scopes=frozenset({"kernel_train"}))
    auditor = Actor(id="a", role=Role.DELEGATED_OPERATOR, scopes=frozenset({"align_audit"}))
    for job_type, spec in specs.items():
        for outsider in (trainer, auditor):
            with pytest.raises(AuthorizationDeniedError):
                service.create_job(job_type, spec, actor=outsider)
        with pytest.raises(AuthenticationRequiredError):
            service.create_job(job_type, spec, actor=ANONYMOUS)
        scoped = Actor(id="s", role=Role.DELEGATED_OPERATOR, scopes=frozenset({job_type.value}))
        assert service.create_job(job_type, spec, actor=scoped).status is JobStatus.QUEUED


def test_alignment_job_specs_are_typed_bounded_and_contained(tmp_path: Path) -> None:
    _settings, service, owner = _service(tmp_path)
    for job_type, bad in (
        (JobType.ALIGN_GENERATE, {"mode": "critique", "checkpoint_path": "."}),
        (JobType.ALIGN_GENERATE, {"mode": "eval-items", "checkpoint_path": "../escape"}),
        (JobType.ALIGN_GENERATE, {"mode": "eval-items", "checkpoint_path": ".", "count": 0}),
        (JobType.ALIGN_GENERATE, {"mode": "anything", "checkpoint_path": "."}),
        (JobType.ALIGN_JUDGE, {"subject": "preference", "checkpoint_path": ".", "pairs": [[H]]}),
        (JobType.ALIGN_JUDGE, {"subject": "eval-item", "checkpoint_path": "/etc"}),
        (JobType.ALIGN_DATASET, {"kind": "sft"}),
        (JobType.ALIGN_DATASET, {"kind": "sft", "tokenizer_id": H, "include": ["everything"]}),
        (JobType.ALIGN_SFT, {"init_path": "."}),
        (JobType.ALIGN_SFT, {"init_path": ".", "dataset_id": H, "logit_teachers": ["../x"]}),
        (JobType.ALIGN_SFT, {"init_path": ".", "dataset_id": H, "strategy": "ddp"}),
        (JobType.ALIGN_PREFERENCE, {"init_path": ".", "dataset_id": H, "beta": 0}),
        (JobType.ALIGN_PREFERENCE, {"init_path": ".", "dataset_id": "x"}),
        (JobType.ALIGN_EVALUATE, {"mode": "g10", "before_path": "."}),
        (JobType.ALIGN_EVALUATE, {"mode": "calibration"}),
    ):
        with pytest.raises(InvalidJobSpecError):
            service.create_job(job_type, bad, actor=owner)
    assert service.list_jobs() == []


def test_the_alignment_pipeline_runs_as_jobs_and_never_audits_itself(lab: Lab) -> None:
    from epor.alignment.audit import AuditLedger
    from epor.alignment.records import AlignmentStore

    assert lab.settings.data_root is not None and lab.settings.safety_root is not None
    store = AlignmentStore(lab.settings.data_root)
    ledger = AuditLedger(lab.settings.safety_root)
    base = lab.base  # type: ignore[attr-defined]
    sampled = lab.job(
        JobType.ALIGN_GENERATE,
        {"mode": "teacher-sample", "checkpoint_path": base, "max_new_tokens": 6},
    )
    assert len(sampled["examples"]) == 8 and sampled["generator"]["kind"] == "epor-checkpoint"
    sft_set = lab.job(
        JobType.ALIGN_DATASET,
        {"kind": "sft", "checkpoint_path": base, "include": ["fixtures", "teacher-outputs"]},
    )
    sft = lab.job(
        JobType.ALIGN_SFT,
        {
            "init_path": base,
            "dataset_id": sft_set["dataset_id"],
            "max_steps": 2,
            "seq_len": 64,
            "tokens_per_batch": 128,
            "precision": "fp32",
            "checkpoint_every": 2,
        },
    )
    assert sft["objective"] == "sft" and "not evidence" in sft["scope"]
    examples = [i for i, r in store.records("example") if r.purpose == "demonstration"][:3]
    critique = lab.job(
        JobType.ALIGN_GENERATE,
        {
            "mode": "critique",
            "checkpoint_path": sft["output_path"],
            "examples": examples,
            "obligations": ["no-harmful-output"],
            "max_new_tokens": 6,
        },
    )
    assert len(critique["candidates"]) == 3
    judged = lab.job(
        JobType.ALIGN_JUDGE,
        {
            "subject": "preference",
            "checkpoint_path": sft["output_path"],
            "pairs": [[c["example"], c["revision"]] for c in critique["candidates"]],
            "min_margin": 0.0,
        },
    )
    assert judged["audit_state"] == "unaudited"
    # A job never creates or completes an audit: the queue is still empty.
    assert ledger.items() == []
    from epor.alignment.audit import sample_run

    sample_run(ledger, store, judged["judging_run"], per_stratum=3, seed=0, actor_id="owner")
    assert ledger.items() and all(item.verdict is None for item in ledger.items())
    calibration = lab.job(
        JobType.ALIGN_EVALUATE, {"mode": "calibration", "judging_run": judged["judging_run"]}
    )
    assert calibration["pending"] == len(ledger.items()) and calibration["audited"] == 0
    for item in ledger.items():
        ledger.record_verdict(
            item.id, "accept" if item.judge_accepted else "reject", actor=lab.owner
        )
    preferences = lab.job(
        JobType.ALIGN_DATASET,
        {
            "kind": "preference",
            "checkpoint_path": sft["output_path"],
            "include": ["fixtures", "ai-judge"],
            "validation_percent": 35,
        },
    )
    from epor.alignment.audit import default_thresholds, run_admission

    admitted = run_admission(ledger, judged["judging_run"], **default_thresholds()).admitted
    # Admission needs audited accepted AND rejected decisions; a run whose judge
    # accepted everything is (correctly) not admitted at all.
    assert admitted == (judged["accepted"] > 0 and judged["rejected"] > 0)
    assert preferences["candidates_admitted"] == (len(judged["preferences"]) if admitted else 0)
    dpo = lab.job(
        JobType.ALIGN_PREFERENCE,
        {
            "init_path": sft["output_path"],
            "dataset_id": preferences["dataset_id"],
            "max_steps": 2,
            "pairs_per_step": 2,
        },
    )
    assert dpo["steps_completed"] == 2
    monitor = lab.job(
        JobType.ALIGN_EVALUATE,
        {
            "mode": "monitor",
            "policy_path": dpo["output_path"],
            "reference_path": sft["output_path"],
            "dataset_id": preferences["dataset_id"],
            "max_new_tokens": 6,
        },
    )
    assert set(monitor["monitor_status"]) == {"kl", "proxy_minus_audited", "length_ratio"}
    g10 = lab.job(
        JobType.ALIGN_EVALUATE,
        {
            "mode": "g10",
            "before_path": sft["output_path"],
            "after_path": dpo["output_path"],
            "max_new_tokens": 6,
        },
    )
    assert g10["gate"] in {"blocked", "escalated", "no-regression-observed"}
    g10_items = [item for item in ledger.items() if item.subject == "g10-item"]
    assert len(g10_items) == g10["failing_items"]
    assert all(item.verdict is None for item in g10_items)

    from epor.control.training_runs import training_summary

    assert lab.settings.artifact_root is not None
    runs = training_summary(lab.settings.artifact_root, lab.settings.project_root)
    objectives = {run["objective"] for run in runs["runs"]}
    assert {"pretrain", "sft", "dpo"} <= objectives and runs["integrity_errors"] == []


def test_fine_tune_job_cancellation_is_cooperative(lab: Lab) -> None:
    from epor.alignment.datasets import build_sft_dataset
    from epor.alignment.records import AlignmentStore
    from epor.tokenizer import load_tokenizer

    assert lab.settings.data_root is not None
    store = AlignmentStore(lab.settings.data_root)
    demos = [i for i, r in store.records("example") if r.purpose == "demonstration"]
    dataset_id, _ = build_sft_dataset(
        store, load_tokenizer(lab.settings.data_root, lab.shards.tokenizer_id), demos
    )
    job = lab.service.create_job(
        JobType.ALIGN_SFT,
        {
            "init_path": lab.base,  # type: ignore[attr-defined]
            "dataset_id": dataset_id,
            "max_steps": 5_000,
            "seq_len": 64,
            "tokens_per_batch": 128,
            "precision": "fp32",
            "checkpoint_every": 10_000,
            "validation": False,
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
    assert lab.settings.artifact_root is not None
    from epor.training.run_checkpoint import latest_checkpoint

    assert latest_checkpoint(lab.settings.artifact_root / job.id / "sft") is not None


# --- the audit and report API -----------------------------------------------------------


def _queue_one(harness: Harness) -> tuple[str, str]:
    """A synthetic judging run with one queued accepted decision."""

    from epor.alignment.audit import AuditLedger
    from epor.alignment.records import AlignmentStore

    assert harness.settings.safety_root is not None and harness.settings.data_root is not None
    AlignmentStore(harness.settings.data_root)
    ledger = AuditLedger(harness.settings.safety_root)
    item_id = ledger.enqueue(
        subject="preference",
        judging_run="a" * 64,
        record="b" * 64,
        stratum="accepted",
        actor_id="owner",
        judge_accepted=True,
        confidence=0.9,
        detail={"replies": ["one", "two"], "note": "api_key=abcd1234efgh5678"},
    )
    assert item_id is not None
    return item_id, "a" * 64


def test_verdicts_need_a_verified_human_with_the_audit_scope(harness: Harness) -> None:
    item_id, run_id = _queue_one(harness)
    auditor = harness.delegate("align_audit", label="auditor")
    trainer = harness.delegate("align_sft", "align_preference", "align_judge", label="trainer")
    captured: dict[str, Any] = {}

    async def scenario() -> None:
        url = f"/api/v1/alignment/audits/{item_id}/verdict"
        async with harness.client() as client:
            captured["anonymous"] = await client.post(url, json={"verdict": "accept"})
            captured["listing"] = await client.get("/api/v1/alignment/audits?state=pending")
            captured["bad"] = await client.get("/api/v1/alignment/audits/not-an-id")
            captured["missing"] = await client.get(
                "/api/v1/alignment/audits/00000000-0000-0000-0000-000000000000"
            )
        async with harness.as_bearer(trainer) as client:
            captured["trainer"] = await client.post(url, json={"verdict": "accept"})
        async with harness.as_bearer(auditor) as client:
            captured["invalid"] = await client.post(url, json={"verdict": "maybe"})
            captured["auditor"] = await client.post(
                url, json={"verdict": "reject", "note": "looks wrong"}
            )
            captured["run"] = await client.get(f"/api/v1/alignment/audits?run={run_id}")
            captured["jobs"] = await client.get("/api/v1/jobs")

    run(scenario)
    assert captured["anonymous"].status_code == 401
    listing = captured["listing"].json()
    assert [item["id"] for item in listing] == [item_id]
    assert "api_key=[REDACTED]" in json.dumps(listing)  # values are scanned for credentials
    assert captured["bad"].status_code == 422
    assert captured["missing"].status_code == 404
    assert captured["trainer"].status_code == 403
    assert captured["invalid"].status_code == 422
    assert captured["auditor"].status_code == 200
    view = captured["auditor"].json()
    assert view["state"] == "audited" and view["verdict"] == "reject"
    assert view["verdicts"][0]["actor_role"] == "delegated_operator"
    assert view["verdicts"][0]["note"] == "looks wrong"
    assert captured["run"].json()[0]["verdict"] == "reject"
    assert captured["jobs"].json()["count"] == 0  # a verdict is never a job
    decisions = [
        record for record in harness.store.records() if record.get("action_id") == "align_audit"
    ]
    assert {record["outcome"] for record in decisions} >= {"allow", "refuse"}


def test_alignment_summary_and_reports_are_verified(harness: Harness) -> None:
    from epor.alignment.monitors import store_report
    from epor.alignment.records import AlignmentStore

    assert harness.settings.data_root is not None
    store = AlignmentStore(harness.settings.data_root)
    report_id = store_report(
        store, "calibration", {"schema_version": 1, "kind": "calibration", "audited": 0}
    )
    captured: dict[str, httpx.Response] = {}

    async def fetch(*paths: str) -> None:
        async with harness.client() as client:
            for path in paths:
                captured[path] = await client.get(f"/api/v1{path}")

    report_path = f"/alignment/reports/calibration/{report_id}"
    run(lambda: fetch("/alignment", report_path, "/alignment/reports/other/" + H))
    summary = captured["/alignment"].json()
    assert captured["/alignment"].status_code == 200
    assert "no record, run, or report here is evidence" in summary["scope"]
    assert summary["integrity_errors"] == []
    assert captured[report_path].json()["audited"] == 0
    assert captured["/alignment/reports/other/" + H].status_code == 422
    stored = store.data.alignment_path(f"reports/calibration/{report_id}.json")
    stored.chmod(0o600)
    stored.write_bytes(stored.read_bytes().replace(b'"audited":0', b'"audited":9'))
    captured.clear()
    run(lambda: fetch(report_path))
    assert captured[report_path].status_code == 409


def test_alignment_job_results_need_no_new_redaction_allowlist() -> None:
    from epor.control.security import redact

    # Every alignment summary key was chosen to avoid the credential pattern;
    # the only token-named keys a result can carry were allowlisted before.
    result = {
        "dataset_id": H,
        "target_positions": {"train": 3},
        "tokenizer_id": H,
        "evaluation": {"target_tokens": 4},
        "max_new": 6,
        "monitor_status": {"kl": "ok", "proxy_minus_audited": "unknown", "length_ratio": "ok"},
    }
    assert redact(result) == result
    # The stored monitor name would be redacted in a job result, so results use "kl".
    assert redact({"kl_per_token": "ok"}) == {"kl_per_token": "[REDACTED]"}
    assert redact({"output_token_ids": [1]}) == {"output_token_ids": "[REDACTED]"}
