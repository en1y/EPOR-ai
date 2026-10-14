"""Teachers: local EPOR checkpoints and operator-registered teacher-output files.

Two kinds of teacher exist, and neither ever contributes weights:

* **EPOR checkpoints.**  :func:`sample_teacher` runs a local, hash-verified
  kernel checkpoint over prompts and records each exact sampled output (text
  and token IDs) with the checkpoint's identity, the prompt template's hash,
  the sampling settings, the time, and the lineage of every prompt.  These are
  the only teachers whose logits may be distilled, and only into a student
  that shares their tokenizer (:mod:`epor.training.kernel`).
* **Registered output files.**  An operator who holds rights to another
  model's outputs registers the local file twice: once with the data engine
  (a :class:`~epor.data.models.SourceRegistration` that must allow ``train``)
  and once as a :class:`~epor.alignment.records.TeacherSource` stating the
  provider, model, version, licence, that the licence permits training on its
  outputs, and the cost basis — a paid source names its cost-approval record.
  :func:`import_teacher_outputs` then reads the file; EPOR never calls a model
  API, paid or otherwise, and never downloads anything.  External outputs can
  only be used for response (sequence-level) distillation.

Operator-registered *training sources* (curated demonstrations or preference
pairs rather than teacher outputs) enter through :func:`import_registered_source`
under the same data-engine registration rules.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml

from epor.data.models import SourceRegistration
from epor.data.store import canonical_json_bytes, sha256_bytes, sha256_model

from .chat import CHAT_FORMAT_SHA256, ChatMessage
from .records import (
    LOCAL_CHECKPOINT_RIGHTS,
    AlignmentRecordError,
    AlignmentStore,
    ExampleRecord,
    Generator,
    Lineage,
    PreferenceRecord,
    Rights,
    Sampling,
    TeacherSource,
)

TEACHER_TEMPLATE = {
    "id": "epor-teacher-chat-v1",
    "format": CHAT_FORMAT_SHA256,
    "rule": "the prompt conversation is encoded as-is and the reply follows <|assistant|>",
}
TEACHER_TEMPLATE_SHA256 = sha256_bytes(canonical_json_bytes(TEACHER_TEMPLATE))
MAX_TEACHER_FILE_BYTES = 64 * 1024 * 1024


class TeacherRefusedError(PermissionError):
    """A teacher or training source is unregistered, rights-ineligible, or removed."""


def _now() -> datetime:
    return datetime.now(UTC)


def sample_teacher(
    store: AlignmentStore,
    teacher: Any,
    prompts: Sequence[tuple[Sequence[ChatMessage], Sequence[Lineage]]],
    sampling: Sampling,
    *,
    should_cancel: Callable[[], bool] | None = None,
) -> list[str]:
    """Sample one reply per prompt from a local :class:`~epor.alignment.model.ChatModel`."""

    written: list[str] = []
    for index, (messages, lineage) in enumerate(prompts):
        if should_cancel is not None and should_cancel():
            raise InterruptedError("teacher sampling cancelled")
        text, token_ids = teacher.generate(
            messages, sampling.model_copy(update={"seed": (sampling.seed or 0) + index})
        )
        record = ExampleRecord(
            purpose="teacher-output",
            candidate=False,
            generator=teacher.generator,
            rights=LOCAL_CHECKPOINT_RIGHTS,
            prompt_template_id=TEACHER_TEMPLATE["id"],
            prompt_template_sha256=TEACHER_TEMPLATE_SHA256,
            sampling=sampling.model_copy(update={"seed": (sampling.seed or 0) + index}),
            messages=tuple(messages),
            output=text,
            output_token_ids=token_ids,
            created_at=_now(),
            lineage=(*lineage, teacher.lineage()),
        )
        written.append(store.put(record))
    return written


def _training_registration(store: AlignmentStore, source_id: str) -> tuple[SourceRegistration, str]:
    try:
        registration = store.data.registration(source_id)
    except Exception as exc:
        raise TeacherRefusedError(
            f"source {source_id!r} has no data-engine registration; register its rights first"
        ) from exc
    if "train" not in registration.allowed_uses:
        raise TeacherRefusedError(
            f"source {source_id!r} is not registered for training use "
            f"(allowed uses: {', '.join(registration.allowed_uses)})"
        )
    if registration.media_type != "application/jsonl":
        raise TeacherRefusedError(f"source {source_id!r} must be registered as application/jsonl")
    if any(
        item.target_kind == "source_id" and item.target == source_id
        for item in store.data.tombstones()
    ):
        raise TeacherRefusedError(f"source {source_id!r} has been removed (tombstoned)")
    return registration, sha256_model(registration)


def register_teacher(store: AlignmentStore, path: Path) -> tuple[str, TeacherSource]:
    """Record a reviewed teacher-output registration (YAML/JSON) after checking its source."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("a teacher registration must be a mapping")
    try:
        existing = store.teacher_source(str(raw.get("id")))
    except AlignmentRecordError:
        existing = None
    _, registration_sha256 = _training_registration(store, str(raw.get("source_id")))
    rights = dict(raw.get("rights") or {})
    rights["registration_sha256"] = registration_sha256
    teacher = TeacherSource.model_validate({**raw, "rights": rights, "created_at": _now()})
    if existing is not None:
        raise TeacherRefusedError(
            f"teacher {teacher.id!r} is already registered; registrations are immutable"
        )
    return store.put(teacher), teacher


def _read_jsonl(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    if path.is_symlink() or not path.is_file():
        raise TeacherRefusedError("teacher and source files must be regular non-symlink files")
    if path.stat().st_size > MAX_TEACHER_FILE_BYTES:
        raise TeacherRefusedError(f"file exceeds the {MAX_TEACHER_FILE_BYTES}-byte limit")
    payload = path.read_bytes()
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"line {number}: each line must be a JSON object")
        rows.append(value)
    if not rows:
        raise ValueError("file holds no records")
    return payload, rows


def _check_raw(store: AlignmentStore, raw_sha256: str) -> None:
    if any(
        item.target_kind == "raw_sha256" and item.target == raw_sha256
        for item in store.data.tombstones()
    ):
        raise TeacherRefusedError("this file's content has been removed (tombstoned)")


def import_teacher_outputs(store: AlignmentStore, teacher_id: str, path: Path) -> list[str]:
    """Import ``{"messages": [...], "output": str, "sampled_at": iso, ...}`` lines.

    Optional per-line ``sampling`` overrides the registration's settings;
    optional ``prompt_lineage`` (a list of :class:`Lineage` objects) records
    where the prompt material came from.
    """

    teacher_record_id, teacher = store.teacher_source(teacher_id)
    _training_registration(store, teacher.source_id)
    payload, rows = _read_jsonl(path)
    raw_sha256 = sha256_bytes(payload)
    _check_raw(store, raw_sha256)
    generator = Generator(
        kind="registered-output-file",
        provider=teacher.provider,
        model=teacher.model,
        version=teacher.model_version,
    )
    written: list[str] = []
    for number, row in enumerate(rows, start=1):
        try:
            sampled_at = datetime.fromisoformat(str(row["sampled_at"]).replace("Z", "+00:00"))
            sampling = (
                Sampling.model_validate(row["sampling"]) if "sampling" in row else teacher.sampling
            )
            record = ExampleRecord(
                purpose="teacher-output",
                candidate=False,
                generator=generator,
                rights=teacher.rights,
                prompt_template_id=teacher.prompt_template_id,
                prompt_template_sha256=teacher.prompt_template_sha256,
                sampling=sampling,
                messages=tuple(ChatMessage.model_validate(item) for item in row["messages"]),
                output=str(row["output"]),
                created_at=sampled_at,
                lineage=(
                    *(Lineage.model_validate(item) for item in row.get("prompt_lineage", [])),
                    Lineage(kind="source_id", id=teacher.source_id),
                    Lineage(kind="raw_sha256", id=raw_sha256, sha256=raw_sha256),
                    Lineage(
                        kind="alignment_record", id=teacher_record_id, sha256=teacher_record_id
                    ),
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path.name} line {number}: {exc}") from exc
        written.append(store.put(record))
    return written


SourceFormat = Literal["epor-chat-jsonl", "epor-preference-jsonl"]


def import_registered_source(
    store: AlignmentStore, source_id: str, path: Path, fmt: SourceFormat
) -> list[str]:
    """Import operator-registered demonstrations or preference pairs.

    ``epor-chat-jsonl`` lines are ``{"messages": [...], "response": str}``;
    ``epor-preference-jsonl`` lines are ``{"messages": [...], "chosen": str,
    "rejected": str}``.  The data-engine registration must allow training.
    """

    registration, registration_sha256 = _training_registration(store, source_id)
    payload, rows = _read_jsonl(path)
    raw_sha256 = sha256_bytes(payload)
    _check_raw(store, raw_sha256)
    rights = Rights(
        license_id=registration.license_id,
        training_on_outputs_permitted=True,
        permission_basis=registration.permission_basis,
        registration_sha256=registration_sha256,
    )
    lineage = (
        Lineage(kind="source_id", id=source_id, sha256=registration_sha256),
        Lineage(kind="raw_sha256", id=raw_sha256, sha256=raw_sha256),
    )
    generator = Generator(
        kind="registered-source",
        provider=registration.owner_or_steward[:300],
        model=registration.title[:300],
        version=registration_sha256[:12],
    )
    written: list[str] = []
    for number, row in enumerate(rows, start=1):
        try:
            messages = tuple(ChatMessage.model_validate(item) for item in row["messages"])
            if fmt == "epor-chat-jsonl":
                record: Any = ExampleRecord(
                    purpose="demonstration",
                    candidate=False,
                    generator=generator,
                    rights=rights,
                    prompt_template_id="registered-source-v1",
                    prompt_template_sha256=sha256_bytes(source_id.encode()),
                    sampling=Sampling(decoding="external"),
                    messages=messages,
                    output=str(row["response"]),
                    lineage=lineage,
                )
            else:
                record = PreferenceRecord(
                    origin="registered-source",
                    messages=messages,
                    chosen=str(row["chosen"]),
                    rejected=str(row["rejected"]),
                    lineage=lineage,
                )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path.name} line {number}: {exc}") from exc
        written.append(store.put(record))
    return written


__all__ = [
    "TEACHER_TEMPLATE",
    "TEACHER_TEMPLATE_SHA256",
    "SourceFormat",
    "TeacherRefusedError",
    "import_registered_source",
    "import_teacher_outputs",
    "register_teacher",
    "sample_teacher",
]
