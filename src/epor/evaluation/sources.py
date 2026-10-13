"""Adapters for operator-supplied external benchmark files (never downloaded here).

EPOR does not fetch HumanEval, MBPP, GSM8K, MATH, MMLU, or any other public
benchmark.  An operator who has reviewed a benchmark's rights places a local
JSONL file on disk and registers it with the data engine
(:class:`~epor.data.models.SourceRegistration`).  :func:`load_registered_items`
then refuses unless all of these hold:

* the source ID has a registration in the data store;
* the registration's ``allowed_uses`` includes ``evaluate``;
* the registration's ``media_type`` is ``application/jsonl``;
* no tombstone removes the source ID or the file's raw SHA-256;
* the path is a regular, non-symlink file within the size limit, and (when the
  suite pins one) its SHA-256 matches.

The report records the source ID, the file's SHA-256, and the registration's
SHA-256 — never the local path.

Formats (one JSON object per line; an optional ``id``/``task_id`` names items,
otherwise ``<source_id>:<line>`` is used):

``multiple-choice-jsonl``  ``question``, ``choices`` (list), ``answer`` (index or letter)
``exact-match-jsonl``      ``prompt``, ``answer`` (string) or ``answers`` (list)
``gsm8k-jsonl``            ``question``, ``answer`` (worked solution ending ``#### <n>``)
``competition-math-jsonl`` ``problem``, ``solution`` (final answer in ``\\boxed{}``)
``humaneval-jsonl``        ``task_id``, ``prompt``, ``test``, ``entry_point``
``mbpp-jsonl``             ``task_id``, ``text``, ``test_list`` (+ ``test_setup_code``)
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from epor.data.store import DataStore, DataStoreError, sha256_bytes, sha256_model

from .mathcheck import extract_final_answer
from .report import DataSource
from .tasks import (
    CodeItem,
    ExactMatchItem,
    MathItem,
    MultipleChoiceItem,
    TaskItem,
)

ExternalFormat = Literal[
    "multiple-choice-jsonl",
    "exact-match-jsonl",
    "gsm8k-jsonl",
    "competition-math-jsonl",
    "humaneval-jsonl",
    "mbpp-jsonl",
]
FORMAT_TASK_TYPES: dict[str, str] = {
    "multiple-choice-jsonl": "multiple-choice",
    "exact-match-jsonl": "exact-match",
    "gsm8k-jsonl": "math",
    "competition-math-jsonl": "math",
    "humaneval-jsonl": "code",
    "mbpp-jsonl": "code",
}
DEFAULT_MAX_SOURCE_BYTES = 64 * 1024 * 1024


class SourceRefusedError(PermissionError):
    """An external benchmark source is unregistered, rights-incompatible, or altered."""


@dataclass(frozen=True, slots=True)
class LoadedSource:
    items: list[TaskItem]
    lineage: DataSource


def check_registration(store: DataStore, source_id: str) -> str:
    """Return the registration SHA-256 if the source may be used for evaluation."""

    try:
        registration = store.registration(source_id)
    except DataStoreError as exc:
        raise SourceRefusedError(
            f"benchmark source {source_id!r} is not registered; register it with reviewed "
            "rights before evaluating on it"
        ) from exc
    if "evaluate" not in registration.allowed_uses:
        raise SourceRefusedError(
            f"benchmark source {source_id!r} is not registered for evaluation use "
            f"(allowed uses: {', '.join(registration.allowed_uses)})"
        )
    if registration.media_type != "application/jsonl":
        raise SourceRefusedError(
            f"benchmark source {source_id!r} is registered as {registration.media_type}, "
            "not application/jsonl"
        )
    if any(
        tombstone.target_kind == "source_id" and tombstone.target == source_id
        for tombstone in store.tombstones()
    ):
        raise SourceRefusedError(f"benchmark source {source_id!r} has been removed (tombstoned)")
    return sha256_model(registration)


def _letter_or_index(value: Any, count: int) -> int:
    if isinstance(value, bool):
        raise ValueError("answer must be an index or a letter")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and len(value.strip()) == 1 and value.strip().isalpha():
        return ord(value.strip().upper()) - ord("A")
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError(f"unsupported answer {value!r} for {count} choices")


def _item(record: dict[str, Any], fmt: str, item_id: str) -> TaskItem:
    if fmt == "multiple-choice-jsonl":
        choices = [str(choice) for choice in record["choices"]]
        return MultipleChoiceItem(
            id=item_id,
            question=str(record["question"]),
            choices=choices,
            answer=_letter_or_index(record["answer"], len(choices)),
        )
    if fmt == "exact-match-jsonl":
        answers = record.get("answers", [record.get("answer")])
        return ExactMatchItem(
            id=item_id, prompt=str(record["prompt"]), answers=[str(a) for a in answers]
        )
    if fmt in {"gsm8k-jsonl", "competition-math-jsonl"}:
        problem_key, solution_key = (
            ("question", "answer") if fmt == "gsm8k-jsonl" else ("problem", "solution")
        )
        final = extract_final_answer(str(record[solution_key]))
        if final is None:
            raise ValueError("no final answer could be extracted from the reference solution")
        return MathItem(id=item_id, problem=str(record[problem_key]), answer=final)
    if fmt == "humaneval-jsonl":
        return CodeItem(
            id=item_id,
            prompt=str(record["prompt"]),
            test=str(record["test"]),
            entry_point=str(record["entry_point"]),
        )
    if fmt == "mbpp-jsonl":
        tests = [str(test) for test in record["test_list"]]
        if not tests:
            raise ValueError("MBPP item has no tests")
        setup = str(record.get("test_setup_code") or "")
        return CodeItem(
            id=item_id,
            prompt=f'"""\n{record["text"]}\n{tests[0]}\n"""\n',
            test="\n".join([setup, *tests]).strip() + "\n",
            entry_point=None,
        )
    raise ValueError(f"unsupported external format {fmt!r}")


def parse_jsonl_items(
    payload: bytes,
    fmt: ExternalFormat,
    *,
    source_id: str,
    limit: int | None = None,
) -> list[TaskItem]:
    """Parse JSONL bytes in one of the supported public formats (first ``limit`` items)."""

    items: list[TaskItem] = []
    seen: set[str] = set()
    for number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("each line must be a JSON object")
            raw_id = record.get("id", record.get("task_id"))
            item_id = str(raw_id) if raw_id is not None else f"{source_id}:{number}"
            if item_id in seen:
                raise ValueError(f"duplicate item ID {item_id!r}")
            seen.add(item_id)
            items.append(_item(record, fmt, item_id))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{source_id} line {number}: {exc}") from exc
        if limit is not None and len(items) >= limit:
            break
    if not items:
        raise ValueError(f"{source_id}: no items")
    return items


def load_registered_items(
    store: DataStore,
    source_id: str,
    path: Path,
    fmt: ExternalFormat,
    *,
    expected_sha256: str | None = None,
    limit: int | None = None,
    max_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
    task_ids: Sequence[str] = (),
) -> LoadedSource:
    """Load items from a registered local benchmark file, refusing anything unreviewed."""

    registration_sha256 = check_registration(store, source_id)
    if path.is_symlink() or not path.is_file():
        raise SourceRefusedError("benchmark input must be a regular non-symlink file")
    if path.stat().st_size > max_bytes:
        raise SourceRefusedError(f"benchmark input exceeds the {max_bytes}-byte limit")
    payload = path.read_bytes()
    content_sha256 = sha256_bytes(payload)
    if expected_sha256 is not None and content_sha256 != expected_sha256:
        raise SourceRefusedError("benchmark input does not match the suite's pinned SHA-256")
    if any(
        tombstone.target_kind == "raw_sha256" and tombstone.target == content_sha256
        for tombstone in store.tombstones()
    ):
        raise SourceRefusedError("benchmark input content has been removed (tombstoned)")
    items = parse_jsonl_items(payload, fmt, source_id=source_id, limit=limit)
    return LoadedSource(
        items=items,
        lineage=DataSource(
            kind="registered-source",
            source_id=source_id,
            content_sha256=content_sha256,
            registration_sha256=registration_sha256,
            format=fmt,
            task_ids=list(task_ids),
        ),
    )


__all__ = [
    "DEFAULT_MAX_SOURCE_BYTES",
    "FORMAT_TASK_TYPES",
    "ExternalFormat",
    "LoadedSource",
    "SourceRefusedError",
    "check_registration",
    "load_registered_items",
    "parse_jsonl_items",
]
