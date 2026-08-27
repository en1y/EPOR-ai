from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from epor.data.models import BuildRequest, EvidenceSnapshot, SourceRegistration
from epor.data.service import DataEngine, load_registration
from epor.data.store import DataStoreError

STAMP = datetime(2026, 8, 27, 8, 0, tzinfo=UTC)


def registration(
    source_id: str,
    *,
    allowed_uses: list[str] | None = None,
    group_id: str | None = None,
) -> SourceRegistration:
    return SourceRegistration.model_validate(
        {
            "id": source_id,
            "title": f"Fixture {source_id}",
            "owner_or_steward": "EPOR test suite",
            "canonical_origin": f"fixture://{source_id}",
            "acquisition_method": "generated_fixture",
            "acquired_at": STAMP.isoformat(),
            "media_type": "text/plain",
            "license_id": "fixture-only",
            "permission_basis": "Generated locally for deterministic tests.",
            "allowed_uses": allowed_uses or ["research", "train", "evaluate"],
            "sensitive_content_risk": "low",
            "removal_contact": "owner@example.invalid",
            "group_id": group_id,
        }
    )


def build_request(dataset_id: str = "fixture-v1") -> BuildRequest:
    return BuildRequest(
        dataset_id=dataset_id,
        intended_uses=["offline proxy training tests"],
        prohibited_uses=["identity inference", "production deployment"],
    )


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_example_registration_is_strictly_loadable() -> None:
    example = Path(__file__).parents[1] / "configs/data/source-registration.example.yaml"
    loaded = load_registration(example)
    assert loaded.id == "replace-with-source-id"
    assert loaded.allowed_uses == ["research", "train"]


def test_network_registration_requires_hash_pinned_robots_and_terms() -> None:
    payload = registration("network-source").model_dump(mode="json")
    payload["canonical_origin"] = "https://example.invalid/corpus"
    with pytest.raises(ValidationError, match="robots and terms"):
        SourceRegistration.model_validate(payload)

    evidence = [
        EvidenceSnapshot(
            kind=kind,
            observed_at=STAMP,
            canonical_url=f"https://example.invalid/{kind}",
            sha256="a" * 64,
            status="unknown",
        )
        for kind in ("robots", "terms")
    ]
    payload["evidence"] = [item.model_dump(mode="json") for item in evidence]
    assert SourceRegistration.model_validate(payload).evidence == evidence


def test_ingest_redacts_sensitive_values_and_quarantines_unsafe_content(tmp_path: Path) -> None:
    engine = DataEngine(tmp_path / "data")
    safe = write(
        tmp_path / "safe.txt",
        "The research document belongs to Alice alice@example.com. "
        "The password=fixture-secret-value must never enter training. "
        "This technical system describes an offline database protocol.\n",
    )
    unsafe = write(
        tmp_path / "unsafe.txt",
        "This inert fixture names EICAR-STANDARD-ANTIVIRUS-TEST-FILE for quarantine.\n",
    )

    result = engine.ingest(registration("rights-cleared"), [safe, unsafe])
    records = engine.store.documents()

    assert result["admitted"] == 1
    assert result["quarantined"] == 1
    safe_record = next(item for item in records if item.relative_input_name == "safe.txt")
    unsafe_record = next(item for item in records if item.relative_input_name == "unsafe.txt")
    assert "alice@example.com" not in safe_record.text
    assert "fixture-secret-value" not in safe_record.text
    assert "[REDACTED:EMAIL]" in safe_record.text
    assert "[REDACTED:CREDENTIAL]" in safe_record.text
    assert unsafe_record.disposition == "quarantined"
    assert unsafe_record.reason_codes == ["unsafe_content"]
    assert Path(result["audit_path"]).is_file()


def test_registration_is_immutable_and_unknown_rights_are_not_permission(tmp_path: Path) -> None:
    engine = DataEngine(tmp_path / "data")
    source = write(tmp_path / "source.txt", "A sufficiently long local research fixture. " * 8)
    original = registration("restricted", allowed_uses=["research"])
    result = engine.ingest(original, [source])
    assert result["quarantined"] == 1
    assert result["documents"][0]["reason_codes"] == ["rights_not_training_eligible"]

    changed = original.model_copy(update={"license_id": "silently-changed"})
    with pytest.raises(DataStoreError, match="immutable data record"):
        engine.ingest(changed, [source])


def test_global_dedup_splits_and_dataset_card_are_deterministic(tmp_path: Path) -> None:
    engine = DataEngine(tmp_path / "data")
    duplicate_text = (
        "The deterministic corpus document explains a technical protocol and database system. " * 12
    )
    unique_text = (
        "A separate scientific experiment records a theorem equation and hypothesis result. " * 12
    )
    first = write(tmp_path / "first.txt", duplicate_text)
    second = write(tmp_path / "second.txt", duplicate_text.replace("system. ", "system.\n"))
    unique = write(tmp_path / "unique.txt", unique_text)
    engine.ingest(registration("source-a", group_id="family-a"), [first, unique])
    engine.ingest(registration("source-b", group_id="family-a"), [second])

    request = build_request()
    first_build, first_path = engine.build(request)
    second_build, second_path = engine.build(request)

    assert first_build == second_build
    assert first_path == second_path
    assert first_build.counts["exact_duplicates"] == 1
    assert first_build.counts["admitted"] == 2
    assert len({item.split for item in first_build.documents}) == 1
    assert first_build.artifacts["dataset-card.md"]
    card = (first_path.parent / "dataset-card.md").read_text(encoding="utf-8")
    assert "Sources and rights" in card
    assert "Known gaps" in card
    assert "Removal process" in card
    rows = (first_path.parent / f"splits/{first_build.documents[0].split}.jsonl").read_text()
    assert all(json.loads(line)["text"] for line in rows.splitlines())


def test_repository_relative_topology_and_input_order_reach_split_rows(tmp_path: Path) -> None:
    engine = DataEngine(tmp_path / "data")
    dependency = write(
        tmp_path / "dependency.py",
        "def dependency():\n    return 'dependency'\n" * 8,
    )
    consumer = write(tmp_path / "consumer.py", "from package.dependency import dependency\n" * 12)
    code_registration = registration("code-source", group_id="repository-family").model_copy(
        update={"media_type": "text/x-python", "repository_family": "repository-family"}
    )
    engine.ingest(
        code_registration,
        [dependency, consumer],
        relative_names=["package/dependency.py", "package/consumer.py"],
    )

    build, manifest_path = engine.build(build_request("code-v1"))
    split = build.documents[0].split
    rows = [
        json.loads(line)
        for line in (manifest_path.parent / f"splits/{split}.jsonl").read_text().splitlines()
    ]
    assert [row["relative_input_name"] for row in rows] == [
        "package/dependency.py",
        "package/consumer.py",
    ]
    assert [row["source_order"] for row in rows] == [0, 1]


def test_tombstones_propagate_to_future_builds_and_audit_verifies_hashes(
    tmp_path: Path,
) -> None:
    engine = DataEngine(tmp_path / "data")
    removed = write(tmp_path / "removed.txt", "The removable general reference document. " * 20)
    retained = write(tmp_path / "retained.txt", "The retained technical database protocol. " * 20)
    engine.ingest(registration("removal-source"), [removed, retained])
    records = engine.store.documents()
    target = next(item for item in records if item.relative_input_name == "removed.txt")
    engine.remove(
        target_kind="document_id",
        target=target.document_id,
        reason="Fixture removal request",
        requested_by="test-owner",
        requested_at=STAMP,
    )

    build, _ = engine.build(build_request("after-removal"))
    assert target.document_id not in {item.document_id for item in build.documents}
    assert build.counts["tombstoned"] == 1
    assert len(build.tombstone_ids) == 1
    assert engine.audit().integrity_errors == []

    engine.store.raw_path(target.raw_sha256).write_bytes(b"tampered")
    assert any("raw content hash mismatch" in error for error in engine.audit().integrity_errors)
