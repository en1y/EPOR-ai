"""Build content-addressed SFT and preference datasets from provenance records.

Admission rules, applied to every record named by the operator:

* a record removed by a tombstone (directly or through its lineage) is
  excluded and counted;
* a record whose rights do not permit training on its outputs is refused;
* a record from an operator-registered file is re-checked against its
  data-engine registration, which must still allow ``train``;
* a model-authored **candidate** (a critique/revision output or an AI-judged
  preference) is admitted only when ``gate`` — the audit admission rule of
  :mod:`epor.alignment.audit` — admits it, and is otherwise excluded and
  counted.  Without a gate no candidate is admitted.  Critiques are
  intermediate text and never training targets.

Records are split by a salted hash of their ID, so a record keeps its split
across rebuilds.  The manifest pins the chat format, reserved-ID spec,
tokenizer, every split file's hash, every record ID, source counts, the
admission evidence, and exclusion counts; it carries no timestamp, so the same
inputs always give the same dataset ID.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from epor.data.store import canonical_json_bytes, sha256_bytes
from epor.tokenizer.spec import RESERVED_SPEC_SHA256
from epor.training.chat_data import (
    MANIFEST,
    ChatDatasetManifest,
    ChatDatasetSplit,
    DatasetKind,
)

from .chat import (
    CHAT_FORMAT_SHA256,
    CHAT_FORMAT_VERSION,
    ChatMessage,
    encode_conversation,
    encode_prompt,
    encode_response,
)
from .records import AlignmentRecordError, AlignmentStore, ExampleRecord, PreferenceRecord

SPLIT_SALT = "epor-alignment-split-v1"


@dataclass(frozen=True, slots=True)
class Admission:
    admitted: bool
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)


CandidateGate = Callable[[str, Any], Admission]


def _split(record_id: str, validation_percent: int, salt: str) -> str:
    bucket = int(hashlib.sha256(f"{salt}:{record_id}".encode()).hexdigest()[:8], 16) % 100
    return "validation" if bucket < validation_percent else "train"


def _is_candidate(record: Any) -> bool:
    if isinstance(record, ExampleRecord):
        return record.candidate
    return isinstance(record, PreferenceRecord) and record.origin == "ai-judge"


def _check_registered(store: AlignmentStore, record: Any, checked: set[str]) -> None:
    from .teachers import TeacherRefusedError

    for item in record.lineage:
        if item.kind != "source_id" or item.id in checked:
            continue
        registration = store.data.registration(item.id)
        if "train" not in registration.allowed_uses:
            raise TeacherRefusedError(f"source {item.id!r} is no longer registered for training")
        checked.add(item.id)


@dataclass
class _Build:
    rows: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: {"train": [], "validation": []}
    )
    tokens: Counter[str] = field(default_factory=Counter)
    targets: Counter[str] = field(default_factory=Counter)
    sources: Counter[tuple[str, str]] = field(default_factory=Counter)
    excluded: Counter[str] = field(default_factory=Counter)
    admission: list[dict[str, Any]] = field(default_factory=list)


def _select(
    store: AlignmentStore,
    record_ids: Sequence[str],
    expected: type,
    gate: CandidateGate | None,
    build: _Build,
) -> list[tuple[str, Any]]:
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("a dataset must not name a record twice")
    removal = store.removal_index()
    checked: set[str] = set()
    selected: list[tuple[str, Any]] = []
    for identifier in record_ids:
        record = store.get(identifier)
        if not isinstance(record, expected):
            raise AlignmentRecordError(
                f"record {identifier[:12]} is a {record.kind}, not usable here"
            )
        if isinstance(record, ExampleRecord) and record.purpose == "critique":
            raise AlignmentRecordError("critiques are intermediate text and never training targets")
        if removal.removed(identifier):
            build.excluded["removed"] += 1
            continue
        rights = getattr(record, "rights", None)
        if rights is not None and not rights.training_on_outputs_permitted:
            raise AlignmentRecordError(
                f"record {identifier[:12]} does not permit training on its outputs"
            )
        _check_registered(store, record, checked)
        if _is_candidate(record):
            decision = (
                gate(identifier, record)
                if gate is not None
                else Admission(False, "no audit admission rule was applied")
            )
            build.admission.append(
                {
                    "record": identifier,
                    "admitted": decision.admitted,
                    "reason": decision.reason,
                    **decision.evidence,
                }
            )
            if not decision.admitted:
                build.excluded["candidate-not-admitted"] += 1
                continue
        selected.append((identifier, record))
    return selected


def _source_key(record: Any) -> tuple[str, str]:
    if isinstance(record, ExampleRecord):
        return record.generator.kind, record.purpose
    return record.origin, "preference"


def _write(
    store: AlignmentStore,
    kind: DatasetKind,
    tokenizer: Any,
    build: _Build,
    notes: Sequence[str],
) -> tuple[str, ChatDatasetManifest]:
    if not build.rows["train"]:
        raise ValueError("no admissible record landed in the training split")
    files: dict[str, bytes] = {}
    splits: dict[str, ChatDatasetSplit] = {}
    for split, rows in build.rows.items():
        payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
        files[f"{split}.jsonl"] = payload
        splits[split] = ChatDatasetSplit(
            file=f"{split}.jsonl",
            sha256=sha256_bytes(payload),
            examples=len(rows),
            tokens=build.tokens[split],
            target_tokens=build.targets[split],
            record_ids=tuple(row["record"] for row in rows),
        )
    manifest = ChatDatasetManifest(
        kind=kind,
        chat_format=CHAT_FORMAT_VERSION,
        chat_format_sha256=CHAT_FORMAT_SHA256,
        reserved_spec_sha256=RESERVED_SPEC_SHA256,
        tokenizer_id=tokenizer.tokenizer_id,
        vocab_size=tokenizer.vocab_size,
        splits=splits,  # type: ignore[arg-type]
        sources=tuple(
            {"generator": generator, "purpose": purpose, "records": count}
            for (generator, purpose), count in sorted(build.sources.items())
        ),
        admission={"candidates": build.admission},
        excluded=dict(sorted(build.excluded.items())),
        notes=tuple(notes),
    )
    dataset_id = sha256_bytes(canonical_json_bytes(manifest))
    for name, payload in files.items():
        store.data.write_alignment_file(f"datasets/{dataset_id}/{name}", payload)
    store.data.write_alignment_file(
        f"datasets/{dataset_id}/{MANIFEST}", canonical_json_bytes(manifest)
    )
    return dataset_id, manifest


def build_sft_dataset(
    store: AlignmentStore,
    tokenizer: Any,
    record_ids: Sequence[str],
    *,
    validation_percent: int = 10,
    gate: CandidateGate | None = None,
    notes: Sequence[str] = (),
) -> tuple[str, ChatDatasetManifest]:
    """An ``sft`` dataset of example records (demonstrations, teacher outputs, revisions)."""

    if not 0 <= validation_percent < 100:
        raise ValueError("validation_percent must be between 0 and 99")
    build = _Build()
    for identifier, record in _select(store, record_ids, ExampleRecord, gate, build):
        messages = (*record.messages, ChatMessage(role="assistant", content=record.output))
        ids, mask = encode_conversation(tokenizer, messages)
        split = _split(identifier, validation_percent, SPLIT_SALT)
        build.rows[split].append(
            {"record": identifier, "ids": ids, "mask": "".join("1" if m else "0" for m in mask)}
        )
        build.tokens[split] += len(ids)
        build.targets[split] += sum(mask[1:])
        build.sources[_source_key(record)] += 1
    return _write(store, "sft", tokenizer, build, notes)


def build_preference_dataset(
    store: AlignmentStore,
    tokenizer: Any,
    record_ids: Sequence[str],
    *,
    validation_percent: int = 10,
    gate: CandidateGate | None = None,
    notes: Sequence[str] = (),
) -> tuple[str, ChatDatasetManifest]:
    """A ``preference`` dataset of chosen/rejected pairs for DPO."""

    if not 0 <= validation_percent < 100:
        raise ValueError("validation_percent must be between 0 and 99")
    build = _Build()
    for identifier, record in _select(store, record_ids, PreferenceRecord, gate, build):
        prompt = encode_prompt(tokenizer, record.messages)
        chosen = encode_response(tokenizer, record.chosen)
        rejected = encode_response(tokenizer, record.rejected)
        split = _split(identifier, validation_percent, SPLIT_SALT)
        build.rows[split].append(
            {"record": identifier, "prompt": prompt, "chosen": chosen, "rejected": rejected}
        )
        build.tokens[split] += len(prompt) + len(chosen) + len(rejected)
        build.targets[split] += len(chosen) + len(rejected)
        build.sources[_source_key(record)] += 1
    return _write(store, "preference", tokenizer, build, notes)


def describe_dataset(dataset_id: str, manifest: ChatDatasetManifest) -> dict[str, Any]:
    """A small summary whose keys avoid the control plane's credential pattern."""

    return {
        "dataset_id": dataset_id,
        "kind": manifest.kind,
        "examples": {split: item.examples for split, item in manifest.splits.items()},
        "target_positions": {split: item.target_tokens for split, item in manifest.splits.items()},
        "excluded": manifest.excluded,
        "sources": list(manifest.sources),
        "candidates_admitted": sum(
            1 for item in manifest.admission.get("candidates", []) if item["admitted"]
        ),
    }


__all__ = [
    "SPLIT_SALT",
    "Admission",
    "CandidateGate",
    "build_preference_dataset",
    "build_sft_dataset",
    "describe_dataset",
]
