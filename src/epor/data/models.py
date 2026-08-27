"""Versioned contracts for the provenance-first data engine."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Identifier = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,95}$")]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class FrozenSchema(BaseModel):
    """Strict immutable value used in content-addressed records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceSnapshot(FrozenSchema):
    """A hash-pinned robots or terms observation made during registration."""

    kind: Literal["robots", "terms"]
    observed_at: datetime
    canonical_url: str = Field(min_length=1, max_length=2_048)
    sha256: Sha256
    status: Literal["applicable", "not_applicable", "unknown"]
    notes: str = Field(default="", max_length=1_000)

    @field_validator("observed_at")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("evidence timestamps must be timezone-aware")
        return value.astimezone(UTC)


class SourceRegistration(FrozenSchema):
    """Reviewed rights and stewardship record required before acquisition."""

    schema_version: Literal[1] = 1
    id: Identifier
    title: str = Field(min_length=1, max_length=300)
    owner_or_steward: str = Field(min_length=1, max_length=300)
    canonical_origin: str = Field(min_length=1, max_length=2_048)
    acquisition_method: Literal["manual_local", "reviewed_export", "generated_fixture"]
    acquired_at: datetime
    media_type: Literal[
        "text/plain",
        "text/markdown",
        "application/jsonl",
        "text/x-python",
        "text/x-source-code",
    ] = "text/plain"
    license_id: str = Field(min_length=1, max_length=200)
    permission_basis: str = Field(min_length=1, max_length=2_000)
    allowed_uses: list[Literal["research", "train", "evaluate", "redistribute"]] = Field(
        min_length=1,
        max_length=4,
    )
    geographic_constraints: list[str] = Field(default_factory=list, max_length=32)
    contractual_constraints: list[str] = Field(default_factory=list, max_length=32)
    evidence: list[EvidenceSnapshot] = Field(default_factory=list, max_length=8)
    sensitive_content_risk: Literal["low", "moderate", "high", "unknown"]
    removal_contact: str = Field(min_length=1, max_length=500)
    language_hint: str | None = Field(default=None, max_length=40)
    domain_hint: str | None = Field(default=None, max_length=80)
    repository_family: str | None = Field(default=None, max_length=200)
    benchmark_family: str | None = Field(default=None, max_length=200)
    group_id: str | None = Field(default=None, max_length=200)
    notes: str = Field(default="", max_length=4_000)

    @field_validator("acquired_at")
    @classmethod
    def acquisition_timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("acquisition timestamp must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator(
        "geographic_constraints",
        "contractual_constraints",
        "allowed_uses",
    )
    @classmethod
    def unique_lists(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("registration lists must not contain duplicates")
        return value

    @model_validator(mode="after")
    def evidence_is_complete(self) -> SourceRegistration:
        by_kind = {item.kind for item in self.evidence}
        if self.canonical_origin.startswith(("http://", "https://")) and by_kind != {
            "robots",
            "terms",
        }:
            raise ValueError("network origins require both robots and terms evidence records")
        return self


class Classification(FrozenSchema):
    label: str = Field(min_length=1, max_length=80)
    confidence: float = Field(ge=0.0, le=1.0)
    classifier: str = Field(min_length=1, max_length=120)


class Finding(FrozenSchema):
    kind: str = Field(min_length=1, max_length=96)
    count: int = Field(ge=1)
    action: Literal["redacted", "quarantined", "reported"]


class DocumentRecord(FrozenSchema):
    """Normalized document plus bounded policy findings and parent hashes."""

    schema_version: Literal[1] = 1
    document_id: Sha256
    source_id: Identifier
    relative_input_name: str = Field(min_length=1, max_length=1_024)
    source_order: int = Field(ge=0)
    media_type: str = Field(min_length=1, max_length=160)
    acquired_at: datetime
    raw_sha256: Sha256
    raw_size_bytes: int = Field(ge=0)
    registration_sha256: Sha256
    normalized_sha256: Sha256
    normalized_size_bytes: int = Field(ge=0)
    text: str
    language: Classification
    domain: Classification
    quality: Classification
    group_id: str = Field(min_length=1, max_length=200)
    repository_family: str | None = Field(default=None, max_length=200)
    benchmark_family: str | None = Field(default=None, max_length=200)
    findings: list[Finding] = Field(default_factory=list)
    disposition: Literal["admitted", "quarantined"]
    reason_codes: list[str] = Field(default_factory=list)

    @field_validator("acquired_at")
    @classmethod
    def record_timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("document timestamps must be timezone-aware")
        return value.astimezone(UTC)


class Tombstone(FrozenSchema):
    schema_version: Literal[1] = 1
    id: Sha256
    target_kind: Literal["source_id", "document_id", "raw_sha256"]
    target: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=2_000)
    requested_at: datetime
    requested_by: str = Field(min_length=1, max_length=200)
    removal_contact: str | None = Field(default=None, max_length=500)

    @field_validator("requested_at")
    @classmethod
    def tombstone_timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("tombstone timestamp must be timezone-aware")
        return value.astimezone(UTC)


class SplitRatios(FrozenSchema):
    train: int = Field(default=98, ge=0, le=100)
    validation: int = Field(default=1, ge=0, le=100)
    test: int = Field(default=1, ge=0, le=100)

    @model_validator(mode="after")
    def totals_one_hundred(self) -> SplitRatios:
        if self.train + self.validation + self.test != 100:
            raise ValueError("split ratios must total 100")
        if self.train == 0:
            raise ValueError("the training split must be non-empty by policy")
        return self


class BuildRequest(FrozenSchema):
    schema_version: Literal[1] = 1
    dataset_id: Identifier
    source_ids: list[Identifier] | None = Field(default=None, max_length=1_000)
    split_ratios: SplitRatios = Field(default_factory=SplitRatios)
    split_salt: str = Field(default="epor-data-split-v1", min_length=1, max_length=200)
    near_duplicate_hamming_distance: int = Field(default=3, ge=0, le=8)
    intended_uses: list[str] = Field(min_length=1, max_length=32)
    prohibited_uses: list[str] = Field(min_length=1, max_length=32)

    @field_validator("source_ids", "intended_uses", "prohibited_uses")
    @classmethod
    def unique_build_lists(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and len(value) != len(set(value)):
            raise ValueError("build lists must not contain duplicates")
        return value


class DatasetDocument(FrozenSchema):
    document_id: Sha256
    source_id: Identifier
    relative_input_name: str
    source_order: int
    normalized_sha256: Sha256
    split: Literal["train", "validation", "test"]
    group_id: str
    language: str
    domain: str
    quality: float


class DuplicateRecord(FrozenSchema):
    duplicate_document_id: Sha256
    canonical_document_id: Sha256
    kind: Literal["exact", "near"]
    distance: int = Field(ge=0, le=64)


class DatasetBuild(FrozenSchema):
    schema_version: Literal[1] = 1
    pipeline_version: Literal["epor-data-v1"] = "epor-data-v1"
    dataset_id: Identifier
    build_id: Sha256
    deterministic_timestamp: datetime
    policy_sha256: Sha256
    parent_sha256s: list[Sha256]
    source_ids: list[Identifier]
    counts: dict[str, int]
    split_counts: dict[str, int]
    language_counts: dict[str, int]
    domain_counts: dict[str, int]
    rights_counts: dict[str, int]
    finding_counts: dict[str, int]
    documents: list[DatasetDocument]
    duplicates: list[DuplicateRecord]
    tombstone_ids: list[Sha256]
    artifacts: dict[str, Sha256]
    known_gaps: list[str]

    @field_validator("deterministic_timestamp")
    @classmethod
    def build_timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("build timestamp must be timezone-aware")
        return value.astimezone(UTC)


class DataAudit(FrozenSchema):
    schema_version: Literal[1] = 1
    registrations: int
    documents: int
    admitted: int
    quarantined: int
    tombstones: int
    builds: int
    integrity_errors: list[str]
    latest_builds: dict[str, str]
