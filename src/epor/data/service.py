"""High-level rights-aware ingestion, removal, build, and audit service."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .models import (
    BuildRequest,
    DataAudit,
    DatasetBuild,
    DatasetDocument,
    DocumentRecord,
    DuplicateRecord,
    SourceRegistration,
    Tombstone,
)
from .pipeline import (
    PIPELINE_VERSION,
    exact_fingerprint,
    hamming_distance,
    make_document_record,
    simhash64,
    split_for,
)
from .store import (
    DEFAULT_MAX_INPUT_BYTES,
    DataStore,
    canonical_json_bytes,
    sha256_bytes,
    sha256_model,
)

Progress = Callable[[float, str, dict[str, Any] | None], None]


def load_registration(path: Path) -> SourceRegistration:
    """Load one strict reviewed YAML/JSON registration."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("source registration must be a mapping")
    return SourceRegistration.model_validate(raw)


class DataEngine:
    """Deterministic v0.0.3 data engine over a private immutable store."""

    def __init__(self, root: Path) -> None:
        self.store = DataStore(root)

    def ingest(
        self,
        registration: SourceRegistration,
        inputs: Iterable[Path],
        *,
        max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
        check_cancelled: Callable[[], None] | None = None,
        progress: Progress | None = None,
    ) -> dict[str, Any]:
        paths = list(inputs)
        if not paths:
            raise ValueError("at least one local input is required")
        registration_sha256 = self.store.register(registration)
        documents: list[DocumentRecord] = []
        total = len(paths)
        for index, input_path in enumerate(paths, start=1):
            if check_cancelled is not None:
                check_cancelled()
            raw_path, raw_sha256, raw_size = self.store.stream_raw(
                input_path,
                max_bytes=max_input_bytes,
                check_cancelled=check_cancelled,
            )
            record = make_document_record(
                registration,
                registration_sha256=registration_sha256,
                raw_path=raw_path,
                raw_sha256=raw_sha256,
                raw_size_bytes=raw_size,
                relative_input_name=input_path.name,
            )
            self.store.write_document(record)
            documents.append(record)
            if progress is not None:
                progress(
                    index / total,
                    f"Ingested {index} of {total} inputs",
                    {"document_id": record.document_id, "disposition": record.disposition},
                )

        audit_payload = {
            "schema_version": 1,
            "pipeline_version": PIPELINE_VERSION,
            "source_id": registration.id,
            "registration_sha256": registration_sha256,
            "document_ids": [item.document_id for item in documents],
            "raw_sha256s": [item.raw_sha256 for item in documents],
            "counts": dict(Counter(item.disposition for item in documents)),
            "finding_counts": dict(
                sorted(
                    Counter(
                        finding.kind for item in documents for finding in item.findings
                    ).items()
                )
            ),
        }
        audit_path, audit_sha256 = self.store.write_ingest_audit(audit_payload)
        return {
            "source_id": registration.id,
            "registration_sha256": registration_sha256,
            "documents": [
                {
                    "document_id": item.document_id,
                    "raw_sha256": item.raw_sha256,
                    "disposition": item.disposition,
                    "reason_codes": item.reason_codes,
                }
                for item in documents
            ],
            "count": len(documents),
            "admitted": sum(item.disposition == "admitted" for item in documents),
            "quarantined": sum(item.disposition == "quarantined" for item in documents),
            "audit_path": str(audit_path),
            "audit_sha256": audit_sha256,
        }

    def remove(
        self,
        *,
        target_kind: str,
        target: str,
        reason: str,
        requested_by: str,
        requested_at: datetime | None = None,
        removal_contact: str | None = None,
    ) -> Tombstone:
        timestamp = (requested_at or datetime.now(UTC)).astimezone(UTC)
        core = {
            "schema_version": 1,
            "target_kind": target_kind,
            "target": target,
            "reason": reason,
            "requested_at": timestamp.isoformat(),
            "requested_by": requested_by,
            "removal_contact": removal_contact,
        }
        tombstone = Tombstone(id=sha256_bytes(canonical_json_bytes(core)), **core)
        self.store.write_tombstone(tombstone)
        return tombstone

    @staticmethod
    def _tombstoned(record: DocumentRecord, tombstones: list[Tombstone]) -> bool:
        values = {
            "source_id": record.source_id,
            "document_id": record.document_id,
            "raw_sha256": record.raw_sha256,
        }
        return any(values[item.target_kind] == item.target for item in tombstones)

    @staticmethod
    def _deduplicate(
        records: list[DocumentRecord],
        maximum_distance: int,
    ) -> tuple[list[DocumentRecord], list[DuplicateRecord]]:
        kept: list[DocumentRecord] = []
        duplicates: list[DuplicateRecord] = []
        exact: dict[str, DocumentRecord] = {}
        signatures: dict[str, int] = {}
        bands: dict[tuple[int, int], set[str]] = defaultdict(set)

        for record in sorted(records, key=lambda item: item.document_id):
            fingerprint = exact_fingerprint(record.text)
            exact_match = exact.get(fingerprint)
            if exact_match is not None:
                duplicates.append(
                    DuplicateRecord(
                        duplicate_document_id=record.document_id,
                        canonical_document_id=exact_match.document_id,
                        kind="exact",
                        distance=0,
                    )
                )
                continue

            signature = simhash64(record.text)
            candidates: set[str] = set()
            for band in range(4):
                candidates.update(bands[(band, (signature >> (band * 16)) & 0xFFFF)])
            nearest: tuple[int, str] | None = None
            for document_id in sorted(candidates):
                distance = hamming_distance(signature, signatures[document_id])
                if distance <= maximum_distance and (
                    nearest is None or (distance, document_id) < nearest
                ):
                    nearest = (distance, document_id)
            if nearest is not None:
                duplicates.append(
                    DuplicateRecord(
                        duplicate_document_id=record.document_id,
                        canonical_document_id=nearest[1],
                        kind="near",
                        distance=nearest[0],
                    )
                )
                continue

            kept.append(record)
            exact[fingerprint] = record
            signatures[record.document_id] = signature
            for band in range(4):
                bands[(band, (signature >> (band * 16)) & 0xFFFF)].add(record.document_id)
        return kept, duplicates

    def build(
        self,
        request: BuildRequest,
        *,
        check_cancelled: Callable[[], None] | None = None,
        progress: Progress | None = None,
    ) -> tuple[DatasetBuild, Path]:
        registrations = {item.id: item for item in self.store.registrations()}
        requested_sources = set(request.source_ids or registrations)
        missing = requested_sources.difference(registrations)
        if missing:
            raise ValueError(f"unregistered data source(s): {', '.join(sorted(missing))}")
        documents = [
            item for item in self.store.documents() if item.source_id in requested_sources
        ]
        tombstones = self.store.tombstones()
        tombstoned = [item for item in documents if self._tombstoned(item, tombstones)]
        candidates = [
            item
            for item in documents
            if item.disposition == "admitted" and not self._tombstoned(item, tombstones)
        ]
        if not candidates:
            raise ValueError("no admitted, non-tombstoned documents are eligible for this build")
        if check_cancelled is not None:
            check_cancelled()
        if progress is not None:
            progress(0.15, "Applying global exact and near-duplicate policy", None)
        kept, duplicates = self._deduplicate(
            candidates,
            request.near_duplicate_hamming_distance,
        )

        dataset_documents: list[DatasetDocument] = []
        by_split: dict[str, list[DocumentRecord]] = {"train": [], "validation": [], "test": []}
        for record in kept:
            split = split_for(
                record.group_id,
                request.split_salt,
                request.split_ratios.train,
                request.split_ratios.validation,
            )
            by_split[split].append(record)
            dataset_documents.append(
                DatasetDocument(
                    document_id=record.document_id,
                    source_id=record.source_id,
                    normalized_sha256=record.normalized_sha256,
                    split=split,  # type: ignore[arg-type]
                    group_id=record.group_id,
                    language=record.language.label,
                    domain=record.domain.label,
                    quality=record.quality.confidence,
                )
            )

        finding_counts = Counter(
            finding.kind for record in documents for finding in record.findings
        )
        language_counts = Counter(item.language.label for item in kept)
        domain_counts = Counter(item.domain.label for item in kept)
        rights_counts = Counter(registrations[item.source_id].license_id for item in kept)
        source_ids = sorted({item.source_id for item in kept})
        registration_hashes = [sha256_model(registrations[item]) for item in source_ids]
        record_hashes = [
            sha256_model(item) for item in sorted(documents, key=lambda x: x.document_id)
        ]
        applied_tombstones = [
            item
            for item in tombstones
            if any(self._tombstoned(record, [item]) for record in documents)
        ]
        parent_hashes = sorted(
            {
                *registration_hashes,
                *record_hashes,
                *(sha256_model(item) for item in applied_tombstones),
            }
        )
        timestamp = max(item.acquired_at for item in kept)
        policy_sha256 = sha256_model(request)
        counts = {
            "registered_documents": len(documents),
            "eligible_before_dedup": len(candidates),
            "admitted": len(kept),
            "quarantined": sum(item.disposition == "quarantined" for item in documents),
            "tombstoned": len(tombstoned),
            "exact_duplicates": sum(item.kind == "exact" for item in duplicates),
            "near_duplicates": sum(item.kind == "near" for item in duplicates),
            "benchmark_markers": finding_counts["benchmark_marker"],
        }
        split_counts = {name: len(items) for name, items in by_split.items()}
        known_gaps = [
            "Language, domain, quality, PII, credential, malware, and benchmark checks are "
            "deterministic offline heuristics; sampled human audit remains required.",
            "Near-duplicate detection uses 64-bit four-gram SimHash and must be revalidated "
            "before target-scale ingestion.",
            "A dataset build records registered benchmark markers but does not prove absence "
            "of unregistered benchmark contamination.",
        ]
        core = {
            "schema_version": 1,
            "pipeline_version": PIPELINE_VERSION,
            "dataset_id": request.dataset_id,
            "deterministic_timestamp": timestamp.isoformat(),
            "policy_sha256": policy_sha256,
            "parent_sha256s": parent_hashes,
            "source_ids": source_ids,
            "counts": counts,
            "split_counts": split_counts,
            "language_counts": dict(sorted(language_counts.items())),
            "domain_counts": dict(sorted(domain_counts.items())),
            "rights_counts": dict(sorted(rights_counts.items())),
            "finding_counts": dict(sorted(finding_counts.items())),
            "documents": [item.model_dump(mode="json") for item in dataset_documents],
            "duplicates": [item.model_dump(mode="json") for item in duplicates],
            "tombstone_ids": sorted(item.id for item in applied_tombstones),
            "known_gaps": known_gaps,
            "intended_uses": request.intended_uses,
            "prohibited_uses": request.prohibited_uses,
        }
        build_id = sha256_bytes(canonical_json_bytes(core))

        split_payloads: dict[str, bytes] = {}
        for name, records in by_split.items():
            rows = []
            for record in sorted(records, key=lambda item: item.document_id):
                rows.append(
                    canonical_json_bytes(
                        {
                            "document_id": record.document_id,
                            "source_id": record.source_id,
                            "text": record.text,
                            "language": record.language.model_dump(mode="json"),
                            "domain": record.domain.model_dump(mode="json"),
                            "quality": record.quality.model_dump(mode="json"),
                            "registration_sha256": record.registration_sha256,
                            "normalized_sha256": record.normalized_sha256,
                            "group_id": record.group_id,
                        }
                    )
                )
            split_payloads[f"splits/{name}.jsonl"] = b"\n".join(rows) + (b"\n" if rows else b"")

        filtered_payload = canonical_json_bytes(
            {
                "schema_version": 1,
                "build_id": build_id,
                "parents": parent_hashes,
                "document_ids": [item.document_id for item in dataset_documents],
                "duplicates": [item.model_dump(mode="json") for item in duplicates],
                "tombstone_ids": sorted(item.id for item in applied_tombstones),
            }
        )
        card = self._dataset_card(
            request=request,
            build_id=build_id,
            timestamp=timestamp,
            source_ids=source_ids,
            registrations=registrations,
            counts=counts,
            split_counts=split_counts,
            language_counts=language_counts,
            domain_counts=domain_counts,
            rights_counts=rights_counts,
            finding_counts=finding_counts,
            known_gaps=known_gaps,
            parent_hashes=parent_hashes,
        ).encode("utf-8")
        artifact_payloads = {
            **split_payloads,
            "filtered.json": filtered_payload,
            "dataset-card.md": card,
        }
        artifacts = {name: sha256_bytes(payload) for name, payload in artifact_payloads.items()}
        manifest = DatasetBuild(
            dataset_id=request.dataset_id,
            build_id=build_id,
            deterministic_timestamp=timestamp,
            policy_sha256=policy_sha256,
            parent_sha256s=parent_hashes,
            source_ids=source_ids,
            counts=counts,
            split_counts=split_counts,
            language_counts=dict(sorted(language_counts.items())),
            domain_counts=dict(sorted(domain_counts.items())),
            rights_counts=dict(sorted(rights_counts.items())),
            finding_counts=dict(sorted(finding_counts.items())),
            documents=dataset_documents,
            duplicates=duplicates,
            tombstone_ids=sorted(item.id for item in applied_tombstones),
            artifacts=artifacts,
            known_gaps=known_gaps,
        )
        if progress is not None:
            progress(0.75, "Writing immutable split shards and dataset card", None)
        for name, payload in artifact_payloads.items():
            if check_cancelled is not None:
                check_cancelled()
            self.store.write_build_file(request.dataset_id, build_id, name, payload)
        manifest_path = self.store.write_build_file(
            request.dataset_id,
            build_id,
            "manifest.json",
            canonical_json_bytes(manifest),
        )
        if progress is not None:
            progress(1.0, "Dataset build complete", {"build_id": build_id})
        return manifest, manifest_path

    @staticmethod
    def _dataset_card(
        *,
        request: BuildRequest,
        build_id: str,
        timestamp: datetime,
        source_ids: list[str],
        registrations: dict[str, SourceRegistration],
        counts: dict[str, int],
        split_counts: dict[str, int],
        language_counts: Counter[str],
        domain_counts: Counter[str],
        rights_counts: Counter[str],
        finding_counts: Counter[str],
        known_gaps: list[str],
        parent_hashes: list[str],
    ) -> str:
        def bullets(items: Iterable[str]) -> str:
            return "\n".join(f"- {item}" for item in items)

        source_lines = [
            f"{source_id}: {registrations[source_id].title} — {registrations[source_id].license_id}"
            for source_id in source_ids
        ]
        return f"""# Dataset card: {request.dataset_id}

- Build ID: `{build_id}`
- Pipeline: `{PIPELINE_VERSION}`
- Deterministic timestamp: `{timestamp.isoformat()}`
- Policy hash: `{sha256_model(request)}`

## Intended uses

{bullets(request.intended_uses)}

## Prohibited uses

{bullets(request.prohibited_uses)}

## Sources and rights

{bullets(source_lines)}

Rights categories: `{json.dumps(dict(sorted(rights_counts.items())), sort_keys=True)}`

## Composition

- Counts: `{json.dumps(counts, sort_keys=True)}`
- Splits: `{json.dumps(split_counts, sort_keys=True)}`
- Languages: `{json.dumps(dict(sorted(language_counts.items())), sort_keys=True)}`
- Domains: `{json.dumps(dict(sorted(domain_counts.items())), sort_keys=True)}`
- Findings: `{json.dumps(dict(sorted(finding_counts.items())), sort_keys=True)}`

Exact and global near-duplicate removal occur before stable content-family split assignment.
PII and credential values are redacted without retaining matched values in ordinary logs.
Unsafe content and executable inputs are quarantined outside the admitted build.

## Removal process

Removal tombstones target a source ID, document ID, or raw SHA-256. Every future build
applies the immutable tombstone set before deduplication and split assignment. Continued
training and released derivatives require a separate documented impact decision.

## Known gaps

{bullets(known_gaps)}

## Immutable parents

{bullets(f'`{item}`' for item in parent_hashes)}
"""

    def audit(self) -> DataAudit:
        return self.store.audit()

    def dataset_summary(self) -> dict[str, Any]:
        audit = self.audit()
        registrations = self.store.registrations()
        return {
            **audit.model_dump(mode="json"),
            "sources": [
                {
                    "id": item.id,
                    "title": item.title,
                    "owner_or_steward": item.owner_or_steward,
                    "license_id": item.license_id,
                    "allowed_uses": item.allowed_uses,
                    "sensitive_content_risk": item.sensitive_content_risk,
                }
                for item in registrations
            ],
        }
