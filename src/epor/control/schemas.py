"""Versioned API and job-spec schemas for the v0.0.1 console."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from epor.reference_policy import REFERENCE_CORPUS_MAX_BYTES

from .models import JobStatus, JobType


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class SystemProbeSpec(StrictSchema):
    pass


class ResearchSyncSpec(StrictSchema):
    offline: bool = True
    source_ids: list[str] | None = Field(default=None, max_length=100)


class DataIngestSpec(StrictSchema):
    registration_path: str = Field(max_length=512)
    input_paths: list[str] = Field(min_length=1, max_length=100)
    max_input_bytes: int = Field(default=256 * 1024 * 1024, ge=1, le=256 * 1024 * 1024)


class DataBuildSpec(StrictSchema):
    dataset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,95}$")
    source_ids: list[str] | None = Field(default=None, max_length=1_000)
    train_percent: int = Field(default=98, ge=1, le=100)
    validation_percent: int = Field(default=1, ge=0, le=100)
    test_percent: int = Field(default=1, ge=0, le=100)
    split_salt: str = Field(default="epor-data-split-v1", min_length=1, max_length=200)
    near_duplicate_hamming_distance: int = Field(default=3, ge=0, le=8)
    intended_uses: list[str] = Field(
        default_factory=lambda: ["offline original-weight language-model training"],
        min_length=1,
        max_length=32,
    )
    prohibited_uses: list[str] = Field(
        default_factory=lambda: ["identity inference", "unreviewed redistribution"],
        min_length=1,
        max_length=32,
    )

    @model_validator(mode="after")
    def split_total(self) -> DataBuildSpec:
        if self.train_percent + self.validation_percent + self.test_percent != 100:
            raise ValueError("data split percentages must total 100")
        return self


class DataRemoveSpec(StrictSchema):
    target_kind: Literal["source_id", "document_id", "raw_sha256"]
    target: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=2_000)
    requested_at: datetime | None = None
    removal_contact: str | None = Field(default=None, max_length=500)


class DataAuditSpec(StrictSchema):
    pass


TinyModelConfigPath = Literal[
    "configs/models/epor-tiny.yaml",
    "configs/models/epor-reference.yaml",
]

# Control and standalone commands share the same v0.0.1 reference-runtime cap.
TINY_CONTROL_CORPUS_MAX_BYTES = REFERENCE_CORPUS_MAX_BYTES


class TinyTrainSpec(StrictSchema):
    config_path: TinyModelConfigPath = Field(
        default="configs/models/epor-tiny.yaml",
        alias="model_config",
        serialization_alias="model_config",
    )
    corpus_path: str = Field(
        default="fixtures/tiny_corpus.txt",
        max_length=512,
        description="Contained fixture corpus; capped at 1 MiB for control jobs.",
    )
    output_name: str = Field(default="tiny-train", pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    max_steps: int = Field(default=20, ge=1, le=1_000)
    seed: int = Field(default=1337, ge=0, le=2**32 - 1)


class TinyEvalSpec(StrictSchema):
    checkpoint_path: str = Field(max_length=512)
    corpus_path: str = Field(
        default="fixtures/tiny_corpus.txt",
        max_length=512,
        description="Contained fixture corpus; capped at 1 MiB for control jobs.",
    )
    config_path: TinyModelConfigPath | None = Field(
        default=None,
        alias="model_config",
        serialization_alias="model_config",
    )
    max_batches: int | None = Field(default=None, ge=1, le=10_000)


JOB_SPEC_MODELS: dict[JobType, type[StrictSchema]] = {
    JobType.SYSTEM_PROBE: SystemProbeSpec,
    JobType.RESEARCH_SYNC: ResearchSyncSpec,
    JobType.DATA_INGEST: DataIngestSpec,
    JobType.DATA_BUILD: DataBuildSpec,
    JobType.DATA_REMOVE: DataRemoveSpec,
    JobType.DATA_AUDIT: DataAuditSpec,
    JobType.TINY_TRAIN: TinyTrainSpec,
    JobType.TINY_EVAL: TinyEvalSpec,
}


class JobCreate(StrictSchema):
    # A bounded string, not a JobType. An action EPOR does not declare must
    # reach the covenant and be turned away by it; rejecting it here as a
    # schema violation would decide the case before the covenant saw it.
    type: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    spec: dict[str, Any] = Field(default_factory=dict)


class SessionCreate(StrictSchema):
    """Exchange a credential for a short browser session. Never stored."""

    token: str = Field(min_length=16, max_length=512)


class OperatorCreate(StrictSchema):
    label: str = Field(min_length=1, max_length=120)
    scopes: list[str] = Field(min_length=1, max_length=64)
    expires_at: datetime


class EscalationDecide(StrictSchema):
    approve: bool
    rationale: str = Field(min_length=1, max_length=2_000)


class OperatorRead(BaseModel):
    """A delegation as anyone may see it. No credential value, no digest."""

    id: str
    role: str
    label: str
    scopes: list[str]
    active: bool
    created_at: datetime
    expires_at: datetime | None
    revoked_at: datetime | None
    rotated_at: datetime | None


class OperatorList(BaseModel):
    items: list[OperatorRead]
    count: int


class IssuedCredentialRead(BaseModel):
    """The one and only response that ever carries a credential value.

    It goes to the owner who just minted it and is never retrievable again;
    only its digest is stored.
    """

    operator: OperatorRead
    token: str
    shown_once: Literal[True] = True


class IdentityRead(BaseModel):
    role: Literal["owner", "delegated_operator", "anonymous"]
    principal_id: str | None = None
    label: str = ""
    scopes: list[str] = Field(default_factory=list)
    authenticated: bool = False
    expires_at: datetime | None = None


class ObligationRead(BaseModel):
    key: str
    statement: str


class PrincipleRead(BaseModel):
    priority: int
    key: str
    title: str
    origin: str
    law: str
    obligations: list[ObligationRead]


class RatificationRead(BaseModel):
    present: bool
    matches_covenant: bool
    ratified_on: str | None = None
    reviewer_role: str | None = None
    roadmap_version: str | None = None
    rationale: str | None = None
    evidence: list[str] = Field(default_factory=list)


class EnforcementCheckpointRead(BaseModel):
    name: str
    description: str


class CovenantRead(BaseModel):
    covenant_id: str
    covenant_version: int
    roadmap_version: str
    effective_date: str
    status: Literal["draft", "ratified"]
    sha256: str
    escalation_confidence_floor: float
    principles: list[PrincipleRead]
    limitations: str
    ratification: RatificationRead
    enforcement_checkpoints: list[EnforcementCheckpointRead]


class AssessmentRead(BaseModel):
    priority: int
    status: str
    confidence: float
    rationale: str
    obligation: str | None = None


class ActionRead(BaseModel):
    action_id: str
    summary: str
    outcome: Literal["allow", "refuse", "escalate"]
    job_type: bool
    cli_commands: list[str]
    assessments: list[AssessmentRead]


class ActionList(BaseModel):
    items: list[ActionRead]
    count: int
    undeclared_behavior: str = (
        "An action with no reviewed declaration is assessed as uncertain against "
        "every principle, so it escalates to a human rather than running."
    )


class EscalationRead(BaseModel):
    """A sanitized escalation. The submitted specification is never included."""

    id: str
    actor_id: str
    actor_role: str
    action_id: str
    request_digest: str
    covenant_sha256: str
    binding_priority: int | None
    reasons: list[str]
    state: Literal["open", "approved", "refused", "consumed", "expired"]
    decided_by: str | None
    rationale: str | None
    created_at: datetime
    decided_at: datetime | None
    approval_expires_at: datetime | None
    consumed_at: datetime | None
    consumed_job_id: str | None


class EscalationList(BaseModel):
    items: list[EscalationRead]
    count: int


class ErrorEnvelope(BaseModel):
    code: str
    message: str
    details: Any = None
    request_id: str


class JobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    type: JobType
    status: JobStatus
    spec: dict[str, Any]
    result: dict[str, Any] | None
    progress: float
    error_code: str | None
    error_message: str | None
    worker_id: str | None
    retry_of_id: str | None
    # Null on jobs admitted before v0.0.2 introduced identity.
    submitted_by_id: str | None = None
    submitted_by_role: str | None = None
    admission_covenant_sha256: str | None = None
    escalation_id: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    heartbeat_at: datetime | None
    cancel_requested_at: datetime | None


class JobList(BaseModel):
    items: list[JobRead]
    count: int


class EventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    sequence: int
    kind: str
    payload: dict[str, Any]
    created_at: datetime


class EventList(BaseModel):
    items: list[EventRead]
    next_after: int


class ArtifactRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    job_id: str
    kind: str
    path: str
    sha256: str
    size_bytes: int
    media_type: str | None
    created_at: datetime


class ArtifactList(BaseModel):
    items: list[ArtifactRead]


class HealthRead(BaseModel):
    status: Literal["ok"] = "ok"
    database: Literal["ok"] = "ok"
    version: str


class CapabilityRead(BaseModel):
    bind_host: str
    allowed_origin: str
    job_types: list[JobType]
    cancellation: bool = True
    event_transport: Literal["polling"] = "polling"
    arbitrary_commands: Literal[False] = False
    arbitrary_url_fetching: Literal[False] = False
    filesystem_browser: Literal[False] = False
    hardware: dict[str, Any]


class JobTypeRead(BaseModel):
    type: JobType
    title: str
    description: str
    schema_: dict[str, Any] = Field(alias="schema", serialization_alias="schema")

    model_config = ConfigDict(populate_by_name=True)


class JobTypeList(BaseModel):
    items: list[JobTypeRead]


class ResearchSourceRead(BaseModel):
    id: str
    title: str
    organization: str
    publication_date: str
    media_type: str
    canonical_url: str
    tags: list[str]
    evidence_classes: list[str]
    integrity: Literal["pinned", "content-pinned", "unpinned"]
    sync_status: Literal["verified", "missing", "error", "unknown"]
    raw_sha256: str | None = None
    tracked_raw_sha256: str | None = None
    tracked_content_sha256: str | None = None
    content_scope: Literal["document", "article"] | None = None
    content_profile: Literal["html-document-text-v1", "html-article-text-v1"] | None = None
    size_bytes: int | None = None
    status_message: str = ""


class ResearchCatalogRead(BaseModel):
    schema_version: int | None
    available: bool
    catalog_path: str = "research/catalog.yaml"
    sources: list[ResearchSourceRead]
    count: int
    error: str | None = None


class DataSourceRead(BaseModel):
    id: str
    title: str
    owner_or_steward: str
    license_id: str
    allowed_uses: list[str]
    sensitive_content_risk: str


class DataSummaryRead(BaseModel):
    schema_version: int
    registrations: int
    documents: int
    admitted: int
    quarantined: int
    tombstones: int
    builds: int
    integrity_errors: list[str]
    latest_builds: dict[str, str]
    sources: list[DataSourceRead]


class DocumentSummary(BaseModel):
    slug: str
    title: str
    group: str


class DocumentList(BaseModel):
    items: list[DocumentSummary]
    count: int


class DocumentRead(DocumentSummary):
    markdown: str


class CertifiedProfileRead(BaseModel):
    profile_id: str
    hardware: str
    max_context: int
    precision: str
    status: Literal["planned", "validated", "certified"]
    ram_gib: float | None = None
    vram_gib: float | None = None
    notes: str | None = None


class ModelFamilyRead(BaseModel):
    display_name: str
    slug: str
    family: Literal["alpha", "beta", "gamma"]
    architecture: str
    total_parameters: int | None
    active_parameters: int | None
    core_parameters: int | None
    configured_max_context: int
    trained_max_context: int
    validated_max_context: int
    operational_default_context: int
    certified_profiles: list[CertifiedProfileRead]
    research_features: list[str]
    status: Literal["planned", "proxy", "trained"] = "planned"
    metrics: Literal[None] = None


class ModelFamilyList(BaseModel):
    items: list[ModelFamilyRead]
    count: int


JobStatusFilter = Annotated[JobStatus | None, Field(default=None)]
