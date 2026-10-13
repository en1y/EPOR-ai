"""Tokenizer jobs through covenant admission, the worker, and the read API."""

from __future__ import annotations

import time
from pathlib import Path
from threading import Event, Thread
from typing import Any

import pytest
from conftest import TOKENIZER_FIXTURE_VOCAB, Harness, make_tokenizer_build, run

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
from epor.control.service import JobService
from epor.control.settings import ControlSettings
from epor.control.worker import JobWorker
from epor.data.models import DatasetBuild
from epor.tokenizer import codec, evaluate_tokenizer, train_tokenizer


def tokenizer_service(tmp_path: Path) -> tuple[ControlSettings, JobService, Actor, DatasetBuild]:
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
    owner = AuthorityStore(settings.safety_root).bootstrap_owner().principal.actor()
    engine = create_sqlite_engine(settings.database_path)
    initialize_database(engine)
    service = JobService(create_session_factory(engine), settings)
    build = make_tokenizer_build(tmp_path)
    return settings, service, owner, build


def run_job(worker: JobWorker, expected: Job) -> Job:
    completed = worker.run_once()
    assert completed is not None
    assert completed.id == expected.id
    return completed


def test_tokenizer_jobs_run_through_admission_worker_and_artifacts(tmp_path: Path) -> None:
    settings, service, owner, build = tokenizer_service(tmp_path)
    worker = JobWorker(settings, service=service, worker_id="tokenizer-worker")
    source = {"dataset_id": build.dataset_id, "build_id": build.build_id}

    train = service.create_job(
        JobType.TOKENIZER_TRAIN,
        {**source, "vocab_size": TOKENIZER_FIXTURE_VOCAB},
        actor=owner,
    )
    trained = run_job(worker, train)
    assert trained.status is JobStatus.SUCCEEDED, trained.error_message
    assert trained.result is not None
    tokenizer_id = trained.result["tokenizer_id"]
    assert trained.result["declared_vocab_size"] == TOKENIZER_FIXTURE_VOCAB
    assert trained.result["learned_vocab_size"] <= TOKENIZER_FIXTURE_VOCAB
    assert [item.media_type for item in service.list_artifacts(train.id)] == ["application/json"]

    shard = service.create_job(
        JobType.TOKENIZER_SHARD,
        {**source, "tokenizer_id": tokenizer_id, "max_shard_tokens": 4096},
        actor=owner,
    )
    sharded = run_job(worker, shard)
    assert sharded.status is JobStatus.SUCCEEDED, sharded.error_message
    assert sharded.result is not None
    assert sharded.result["created"] is True
    assert sharded.result["total_documents"] == len(build.documents)

    repeat = service.create_job(
        JobType.TOKENIZER_SHARD,
        {**source, "tokenizer_id": tokenizer_id, "max_shard_tokens": 4096},
        actor=owner,
    )
    repeated = run_job(worker, repeat)
    assert repeated.result is not None
    assert repeated.result["created"] is False
    assert repeated.result["tokenized_id"] == sharded.result["tokenized_id"]

    evaluate = service.create_job(
        JobType.TOKENIZER_EVALUATE,
        {**source, "tokenizer_id": tokenizer_id},
        actor=owner,
    )
    evaluated = run_job(worker, evaluate)
    assert evaluated.status is JobStatus.SUCCEEDED, evaluated.error_message
    assert evaluated.result is not None
    assert evaluated.result["split"] == "validation"
    assert evaluated.result["in_sample"] is False
    assert evaluated.result["round_trip_failures"] == 0

    in_sample = service.create_job(
        JobType.TOKENIZER_EVALUATE,
        {**source, "tokenizer_id": tokenizer_id, "split": "train"},
        actor=owner,
    )
    flagged = run_job(worker, in_sample)
    assert flagged.result is not None and flagged.result["in_sample"] is True
    assert any(
        event.kind == "job.log" and "in-sample" in str(event.payload)
        for event in service.list_events(in_sample.id)
    )


def test_round_trip_failures_fail_the_job_but_keep_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, service, owner, build = tokenizer_service(tmp_path)
    assert settings.data_root is not None
    trained = train_tokenizer(
        settings.data_root, build.dataset_id, build.build_id, vocab_size=TOKENIZER_FIXTURE_VOCAB
    )
    monkeypatch.setattr(codec.EporTokenizer, "decode_bytes", lambda *_a, **_k: b"corrupted")
    worker = JobWorker(settings, service=service, worker_id="tokenizer-worker")
    job = service.create_job(
        JobType.TOKENIZER_EVALUATE,
        {
            "dataset_id": build.dataset_id,
            "build_id": build.build_id,
            "tokenizer_id": trained.tokenizer_id,
        },
        actor=owner,
    )
    failed = run_job(worker, job)
    assert failed.status is JobStatus.FAILED
    assert "round-trip" in (failed.error_message or "")
    assert len(service.list_artifacts(job.id)) == 1
    assert len(list((settings.data_root / "evaluations").glob("*.json"))) == 1


def test_shard_job_cancels_cooperatively_without_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, service, owner, build = tokenizer_service(tmp_path)
    assert settings.data_root is not None
    trained = train_tokenizer(
        settings.data_root, build.dataset_id, build.build_id, vocab_size=TOKENIZER_FIXTURE_VOCAB
    )
    started = Event()
    real_encode = codec.EporTokenizer.encode

    def slow_encode(self: codec.EporTokenizer, *args: Any, **kwargs: Any) -> Any:
        if not started.is_set():
            started.set()
            time.sleep(0.3)
        return real_encode(self, *args, **kwargs)

    monkeypatch.setattr(codec.EporTokenizer, "encode", slow_encode)
    worker = JobWorker(settings, service=service, worker_id="tokenizer-worker")
    job = service.create_job(
        JobType.TOKENIZER_SHARD,
        {
            "dataset_id": build.dataset_id,
            "build_id": build.build_id,
            "tokenizer_id": trained.tokenizer_id,
        },
        actor=owner,
    )
    thread = Thread(target=worker.run_once)
    thread.start()
    assert started.wait(timeout=5)
    service.cancel(job.id)
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert service.get_job(job.id).status is JobStatus.CANCELLED
    assert list((settings.data_root / "tokenized").iterdir()) == []
    assert list((settings.data_root / "tmp").iterdir()) == []


@pytest.mark.parametrize(
    ("job_type", "spec"),
    [
        (JobType.TOKENIZER_TRAIN, {"vocab_size": 1_024}),
        (JobType.TOKENIZER_SHARD, {"tokenizer_id": "0" * 64}),
        (JobType.TOKENIZER_EVALUATE, {"tokenizer_id": "0" * 64}),
    ],
)
def test_tokenizer_jobs_refuse_unauthorized_actors_before_parsing(
    tmp_path: Path, job_type: JobType, spec: dict[str, Any]
) -> None:
    _settings, service, _owner, build = tokenizer_service(tmp_path)
    full = {"dataset_id": build.dataset_id, "build_id": build.build_id, **spec}
    builder = Actor(id="builder", role=Role.DELEGATED_OPERATOR, scopes=frozenset({"data_build"}))

    with pytest.raises(AuthorizationDeniedError):
        service.create_job(job_type, full, actor=builder)
    with pytest.raises(AuthenticationRequiredError):
        service.create_job(job_type, full, actor=ANONYMOUS)
    assert service.list_jobs() == []

    scoped = Actor(id="tok", role=Role.DELEGATED_OPERATOR, scopes=frozenset({job_type.value}))
    assert service.create_job(job_type, full, actor=scoped).status is JobStatus.QUEUED


def test_tokenizer_job_specs_are_typed_and_bounded(tmp_path: Path) -> None:
    _settings, service, owner, build = tokenizer_service(tmp_path)
    for bad in (
        {"dataset_id": build.dataset_id, "build_id": "../escape"},
        {"dataset_id": build.dataset_id, "build_id": build.build_id, "vocab_size": 70_000},
        {"dataset_id": build.dataset_id, "build_id": build.build_id, "command": "rm -rf /"},
    ):
        with pytest.raises(InvalidJobSpecError):
            service.create_job(JobType.TOKENIZER_TRAIN, bad, actor=owner)
    assert service.list_jobs() == []


def test_tokenizer_summary_lists_verified_artifacts_and_flags_tampering(
    harness: Harness,
) -> None:
    assert harness.settings.data_root is not None
    data_root = harness.settings.data_root
    build = make_tokenizer_build(data_root.parent)
    trained = train_tokenizer(
        data_root, build.dataset_id, build.build_id, vocab_size=TOKENIZER_FIXTURE_VOCAB
    )
    report = evaluate_tokenizer(data_root, trained.tokenizer_id, build.dataset_id, build.build_id)

    summaries: list[dict[str, Any]] = []

    async def fetch() -> None:
        async with harness.client() as client:
            summaries.append((await client.get("/api/v1/tokenizers")).json())

    run(fetch)
    summary = summaries[-1]
    assert summary["integrity_errors"] == []
    [listed] = summary["tokenizers"]
    assert listed["tokenizer_id"] == trained.tokenizer_id
    assert listed["declared_vocab_size"] == TOKENIZER_FIXTURE_VOCAB
    assert listed["learned_vocab_size"] == trained.manifest.learned_vocab_size
    [measured] = summary["fertility_reports"]
    assert measured["report_id"] == report.report_id
    assert measured["in_sample"] is False
    assert "not target-scale" in measured["scope"]
    assert set(measured["by_domain"]) == set(report.by_domain)

    tampered = data_root / "evaluations" / f"{report.report_id}.json"
    tampered.write_bytes(tampered.read_bytes().replace(b'"validation"', b'"test"'))
    run(fetch)
    summary = summaries[-1]
    assert summary["fertility_reports"] == []
    assert len(summary["integrity_errors"]) == 1
