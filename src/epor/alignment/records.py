"""Teacher-data provenance: immutable, content-addressed alignment records.

Every example, judge decision, preference pair, and model-written evaluation
item is a record whose ID is the SHA-256 of its canonical JSON, stored once
beneath ``<data-root>/alignment/records/<kind>/<id>.json`` and never rewritten.
A record carries, as data:

* **who produced it** (:class:`Generator`): the project (authored fixtures), a
  local EPOR checkpoint (by checkpoint, config, and tokenizer hash), an
  operator-registered teacher output file (provider, model, version), an
  operator-registered training source, or a human;
* **the rights it was produced under** (:class:`Rights`): licence, whether
  training on the outputs is permitted and why, and the cost basis.  A paid
  source must name its cost-approval record; EPOR itself never calls one;
* **how it was produced**: prompt-template ID and hash, sampling settings,
  time, the exact sampled output (and, for EPOR teachers, the exact token
  IDs);
* **lineage** (:class:`Lineage`): every source, document, raw file, checkpoint,
  covenant, or parent record whose material entered the prompt or decision.

Removal
-------
Data-engine tombstones propagate here.  A record is removed when a tombstone
names it (``alignment_record``) or anything in its lineage (``source_id``,
``document_id``, ``raw_sha256``, ``alignment_record``), and removal is
transitive through parent records: removing a teacher example removes the
judge decisions about it and the preference pairs built from them.
:class:`RemovalIndex` is the single matching rule; dataset builds apply it and
training refuses a dataset that holds a record removed after it was built.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, TypeAdapter, field_validator, model_validator

from epor.data.models import FrozenSchema, Identifier, Sha256, Tombstone
from epor.data.store import DataStore, DataStoreError, canonical_json_bytes, sha256_bytes

from .chat import ChatMessage

RecordKind = Literal["teacher-source", "example", "judgement", "preference", "eval-item"]
RECORD_KINDS: tuple[RecordKind, ...] = (
    "teacher-source",
    "example",
    "judgement",
    "preference",
    "eval-item",
)
Text = Annotated[str, Field(min_length=1, max_length=16_384)]
ShortText = Annotated[str, Field(min_length=1, max_length=300)]


class AlignmentRecordError(ValueError):
    """An alignment record is missing, altered, rights-ineligible, or removed."""


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        raise ValueError("record timestamps must be timezone-aware")
    return value.astimezone(UTC)


class Generator(FrozenSchema):
    """Who produced a record.  External models appear only as declared names."""

    kind: Literal[
        "project-authored",
        "epor-checkpoint",
        "registered-output-file",
        "registered-source",
        "human",
    ]
    provider: ShortText
    model: ShortText
    version: ShortText
    checkpoint_sha256: Sha256 | None = None
    config_sha256: Sha256 | None = None
    tokenizer_id: Sha256 | None = None

    @model_validator(mode="after")
    def checkpoint_identity(self) -> Generator:
        if self.kind == "epor-checkpoint" and (
            self.checkpoint_sha256 is None or self.tokenizer_id is None
        ):
            raise ValueError("an EPOR checkpoint generator needs its checkpoint and tokenizer hash")
        return self


class Rights(FrozenSchema):
    license_id: ShortText
    training_on_outputs_permitted: bool
    permission_basis: Annotated[str, Field(min_length=1, max_length=2_000)]
    registration_sha256: Sha256 | None = None
    cost: Literal["local-no-cost", "paid"] = "local-no-cost"
    cost_approval: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def paid_needs_approval(self) -> Rights:
        if self.cost == "paid" and not (self.cost_approval or "").strip():
            raise ValueError("a paid teacher source requires a cost-approval record")
        return self


PROJECT_RIGHTS = Rights(
    license_id="EPOR-project-original",
    training_on_outputs_permitted=True,
    permission_basis="Original text authored by the EPOR project for alignment fixtures.",
)
LOCAL_CHECKPOINT_RIGHTS = Rights(
    license_id="EPOR-project-original",
    training_on_outputs_permitted=True,
    permission_basis=(
        "Sampled from an EPOR checkpoint trained locally from scratch on reviewed data; no "
        "external weights or outputs are involved."
    ),
)


class Sampling(FrozenSchema):
    decoding: Literal["authored", "greedy", "temperature", "external"]
    temperature: float | None = Field(default=None, ge=0.0, le=10.0)
    top_k: int | None = Field(default=None, ge=1)
    max_new_tokens: int | None = Field(default=None, ge=0, le=4_096)
    seed: int | None = Field(default=None, ge=0)


AUTHORED = Sampling(decoding="authored")


class Lineage(FrozenSchema):
    kind: Literal[
        "project-fixture",
        "source_id",
        "document_id",
        "raw_sha256",
        "alignment_record",
        "checkpoint",
        "covenant",
    ]
    id: Annotated[str, Field(min_length=1, max_length=200)]
    sha256: Sha256 | None = None


class _Record(FrozenSchema):
    schema_version: Literal[1] = 1
    created_at: datetime | None = None
    lineage: tuple[Lineage, ...] = ()

    @field_validator("created_at")
    @classmethod
    def created_is_aware(cls, value: datetime | None) -> datetime | None:
        return _aware(value)


class TeacherSource(_Record):
    """An operator registration of a local teacher-output file (rights-reviewed).

    ``source_id`` names a data-engine registration, which must allow ``train``;
    this record adds what a teacher needs on top: the model it came from and
    an explicit statement that its licence permits training on its outputs.
    """

    kind: Literal["teacher-source"] = "teacher-source"
    id: Identifier
    source_id: Identifier
    provider: ShortText
    model: ShortText
    model_version: ShortText
    rights: Rights
    prompt_template_id: ShortText
    prompt_template_sha256: Sha256
    sampling: Sampling
    notes: str = Field(default="", max_length=2_000)

    @model_validator(mode="after")
    def outputs_trainable(self) -> TeacherSource:
        if not self.rights.training_on_outputs_permitted:
            raise ValueError(
                "this teacher's licence does not permit training on its outputs; it cannot be "
                "registered as a teacher"
            )
        return self


class CovenantCitation(FrozenSchema):
    """Principles cited by key from ``configs/covenant-v1.yaml`` (never restated)."""

    covenant_sha256: Sha256
    covenant_version: int = Field(ge=1)
    obligations: tuple[Annotated[str, Field(pattern=r"^[a-z][a-z0-9-]*$")], ...] = Field(
        min_length=1
    )


class ExampleRecord(_Record):
    """One demonstration: a prompt conversation and the exact response text.

    ``purpose`` separates project demonstrations, teacher outputs, and
    critique/revision outputs.  ``candidate`` is ``True`` for anything a model
    authored for alignment use; a candidate enters training only through an
    audited judging run (:mod:`epor.alignment.audit`).
    """

    kind: Literal["example"] = "example"
    purpose: Literal["demonstration", "teacher-output", "critique", "revision"]
    candidate: bool
    generator: Generator
    rights: Rights
    prompt_template_id: ShortText
    prompt_template_sha256: Sha256
    sampling: Sampling
    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    output: Annotated[str, Field(max_length=16_384)]
    output_token_ids: tuple[int, ...] | None = None
    covenant: CovenantCitation | None = None

    @model_validator(mode="after")
    def model_output_is_candidate(self) -> ExampleRecord:
        if self.purpose in {"critique", "revision"}:
            if not self.candidate:
                raise ValueError("critique and revision outputs are candidates only")
            if self.covenant is None:
                raise ValueError("critique and revision outputs must cite the covenant")
        if self.messages[-1].role == "assistant":
            raise ValueError("example messages are the prompt; the response is the output")
        return self


class JudgementRecord(_Record):
    """One AI judge decision, stored as data with the same lineage as any example."""

    kind: Literal["judgement"] = "judgement"
    judging_run: Sha256
    subject: Literal["preference", "revision", "eval-item"]
    generator: Generator
    prompt_template_id: ShortText
    prompt_template_sha256: Sha256
    inputs: tuple[Sha256, ...] = Field(min_length=1)
    accepted: bool
    preferred: Sha256 | None = None
    margin: float
    confidence: float = Field(ge=0.0, le=1.0)


class PreferenceRecord(_Record):
    kind: Literal["preference"] = "preference"
    origin: Literal["project-fixture", "registered-source", "ai-judge", "human"]
    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    chosen: Text
    rejected: Text
    judgement: Sha256 | None = None

    @model_validator(mode="after")
    def distinct(self) -> PreferenceRecord:
        if self.chosen == self.rejected:
            raise ValueError("a preference pair needs two different responses")
        if self.origin == "ai-judge" and self.judgement is None:
            raise ValueError("an AI-judged preference must name its judgement record")
        return self


class EvalItemRecord(_Record):
    """A model-written G10 evaluation item: a candidate for human review only."""

    kind: Literal["eval-item"] = "eval-item"
    generator: Generator
    prompt_template_id: ShortText
    prompt_template_sha256: Sha256
    sampling: Sampling
    category: Annotated[str, Field(min_length=1, max_length=80)]
    priority: Literal[1, 2, 3]
    obligation: Annotated[str, Field(pattern=r"^[a-z][a-z0-9-]*$")]
    context: Annotated[str, Field(min_length=1, max_length=600)]
    safe: Annotated[str, Field(min_length=1, max_length=300)]
    unsafe: Annotated[str, Field(min_length=1, max_length=300)]


AlignmentRecord = Annotated[
    TeacherSource | ExampleRecord | JudgementRecord | PreferenceRecord | EvalItemRecord,
    Field(discriminator="kind"),
]
_RECORD = TypeAdapter(AlignmentRecord)


def record_id(record: Any) -> str:
    """Content address of a record: SHA-256 of its canonical JSON."""

    return sha256_bytes(canonical_json_bytes(record))


def parent(identifier: str) -> Lineage:
    """Lineage entry naming a parent alignment record."""

    return Lineage(kind="alignment_record", id=identifier, sha256=identifier)


class AlignmentStore:
    """Immutable alignment records beneath one data root."""

    def __init__(self, data_root: str | Path | DataStore) -> None:
        self.data = data_root if isinstance(data_root, DataStore) else DataStore(Path(data_root))

    @property
    def root(self) -> Path:
        return self.data.alignment_path("records")

    def put(self, record: Any) -> str:
        identifier = record_id(record)
        self.data.write_alignment_file(
            f"records/{record.kind}/{identifier}.json", canonical_json_bytes(record)
        )
        return identifier

    def get(self, identifier: str) -> Any:
        if len(identifier) != 64 or any(c not in "0123456789abcdef" for c in identifier):
            raise AlignmentRecordError(f"not an alignment record ID: {identifier!r}")
        for kind in RECORD_KINDS:
            path = self.root / kind / f"{identifier}.json"
            if path.is_file():
                payload = path.read_bytes()
                if sha256_bytes(payload) != identifier:
                    raise AlignmentRecordError(f"alignment record {identifier} was altered")
                return _RECORD.validate_json(payload)
        raise AlignmentRecordError(f"alignment record not found: {identifier}")

    def ids(self, kind: RecordKind) -> list[str]:
        directory = self.root / kind
        if not directory.is_dir():
            return []
        return sorted(path.stem for path in directory.glob("*.json"))

    def records(self, kind: RecordKind) -> list[tuple[str, Any]]:
        return [(identifier, self.get(identifier)) for identifier in self.ids(kind)]

    def teacher_source(self, teacher_id: str) -> tuple[str, TeacherSource]:
        for identifier, record in self.records("teacher-source"):
            if record.id == teacher_id:
                return identifier, record
        raise AlignmentRecordError(f"teacher source not registered: {teacher_id}")

    def removal_index(self) -> RemovalIndex:
        return RemovalIndex(self, self.data.tombstones())


class RemovalIndex:
    """Tombstone propagation over record lineage, transitive through parents."""

    def __init__(self, store: AlignmentStore, tombstones: Iterable[Tombstone]) -> None:
        self._store = store
        self._targets = {(item.target_kind, item.target) for item in tombstones}
        self._memo: dict[str, bool] = {}

    def _lineage_removed(self, lineage: Sequence[Lineage], seen: frozenset[str]) -> bool:
        for item in lineage:
            if (item.kind, item.id) in self._targets:
                return True
            if item.kind == "alignment_record" and self._removed(item.id, seen):
                return True
        return False

    def _removed(self, identifier: str, seen: frozenset[str]) -> bool:
        if identifier in self._memo:
            return self._memo[identifier]
        if identifier in seen:
            raise AlignmentRecordError("alignment record lineage contains a cycle")
        removed = ("alignment_record", identifier) in self._targets
        if not removed:
            try:
                record = self._store.get(identifier)
            except AlignmentRecordError:
                # A parent that no longer exists cannot vouch for its child.
                removed = True
            else:
                parents = list(record.lineage)
                parents += [parent(value) for value in getattr(record, "inputs", ())]
                if getattr(record, "judgement", None):
                    parents.append(parent(record.judgement))
                if isinstance(record, TeacherSource):
                    parents.append(Lineage(kind="source_id", id=record.source_id))
                removed = self._lineage_removed(parents, seen | {identifier})
        self._memo[identifier] = removed
        return removed

    def removed(self, identifier: str) -> bool:
        return self._removed(identifier, frozenset())


__all__ = [
    "AUTHORED",
    "LOCAL_CHECKPOINT_RIGHTS",
    "PROJECT_RIGHTS",
    "RECORD_KINDS",
    "AlignmentRecord",
    "AlignmentRecordError",
    "AlignmentStore",
    "CovenantCitation",
    "DataStoreError",
    "EvalItemRecord",
    "ExampleRecord",
    "Generator",
    "JudgementRecord",
    "Lineage",
    "PreferenceRecord",
    "RecordKind",
    "RemovalIndex",
    "Rights",
    "Sampling",
    "TeacherSource",
    "parent",
    "record_id",
]
