"""Verified chat datasets (``epor-chat-v1``) as masked training documents.

A chat dataset is built by :mod:`epor.alignment.datasets` from provenance
records and stored beneath ``<data-root>/alignment/datasets/<dataset_id>/``:

``manifest.json``
    :class:`ChatDatasetManifest`; ``dataset_id`` is the SHA-256 of its
    canonical JSON, so the ID pins every split file's hash.
``<split>.jsonl``
    For ``sft`` datasets one object per conversation:
    ``{"record": <id>, "ids": [...], "mask": "0011.."}`` where ``mask[i]`` is
    ``1`` when token ``i`` is a training target.  For ``preference`` datasets
    ``{"record": <id>, "prompt": [...], "chosen": [...], "rejected": [...]}``,
    responses ending with ``<|end_turn|>``.

:class:`ChatCorpus` implements :class:`~epor.training.shard_data.TrainingCorpus`
for ``sft`` datasets, so the v0.0.5 kernel packs and trains them unchanged:
inputs are the conversation tokens, and every target the mask leaves out
(system, user, and tool turns, role tokens) is ``IGNORE_INDEX``.  Opening a
dataset re-verifies every hash and refuses it when any of its records has been
removed since it was built.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor

from epor.data.models import FrozenSchema, Sha256
from epor.data.store import DataStore, canonical_json_bytes, sha256_bytes, sha256_file
from epor.models.config import ModelConfig
from epor.models.reference import IGNORE_INDEX
from epor.tokenizer.corpus import Split
from epor.tokenizer.spec import BOS_ID, EOS_ID, PAD_ID

MANIFEST = "manifest.json"
DatasetKind = Literal["sft", "preference"]


class ChatDatasetError(ValueError):
    """A chat dataset is missing, altered, removed, or incompatible with the model."""


class ChatDatasetSplit(FrozenSchema):
    file: str
    sha256: Sha256
    examples: int = Field(ge=0)
    tokens: int = Field(ge=0)
    target_tokens: int = Field(ge=0)
    record_ids: tuple[Sha256, ...]


class ChatDatasetManifest(FrozenSchema):
    schema_version: Literal[1] = 1
    kind: DatasetKind
    chat_format: str
    chat_format_sha256: Sha256
    reserved_spec_sha256: Sha256
    tokenizer_id: Sha256
    vocab_size: int = Field(ge=1)
    splits: dict[Split, ChatDatasetSplit]
    sources: tuple[dict[str, Any], ...]
    admission: dict[str, Any]
    excluded: dict[str, int]
    notes: tuple[str, ...] = ()


def dataset_directory(data_root: str | Path | DataStore, dataset_id: str) -> Path:
    store = data_root if isinstance(data_root, DataStore) else DataStore(Path(data_root))
    if len(dataset_id) != 64 or any(c not in "0123456789abcdef" for c in dataset_id):
        raise ChatDatasetError(f"not a chat dataset ID: {dataset_id!r}")
    return store.alignment_path(f"datasets/{dataset_id}")


def load_chat_manifest(data_root: str | Path | DataStore, dataset_id: str) -> ChatDatasetManifest:
    path = dataset_directory(data_root, dataset_id) / MANIFEST
    try:
        payload = path.read_bytes()
    except FileNotFoundError as exc:
        raise ChatDatasetError(f"chat dataset not found: {dataset_id}") from exc
    manifest = ChatDatasetManifest.model_validate_json(payload)
    if sha256_bytes(canonical_json_bytes(manifest)) != dataset_id:
        raise ChatDatasetError(f"chat dataset manifest does not match its ID: {dataset_id}")
    return manifest


def read_split_rows(
    data_root: str | Path | DataStore,
    dataset_id: str,
    split: Split,
    *,
    kind: DatasetKind,
    check_removals: bool = True,
) -> tuple[ChatDatasetManifest, list[dict[str, Any]]]:
    """Hash-verified rows of one split, refusing records removed since the build."""

    store = data_root if isinstance(data_root, DataStore) else DataStore(Path(data_root))
    manifest = load_chat_manifest(store, dataset_id)
    if manifest.kind != kind:
        raise ChatDatasetError(f"dataset {dataset_id[:12]} is a {manifest.kind} dataset")
    record = manifest.splits.get(split)
    if record is None:
        raise ChatDatasetError(f"dataset {dataset_id[:12]} has no {split!r} split")
    path = dataset_directory(store, dataset_id) / record.file
    if not path.is_file() or sha256_file(path) != record.sha256:
        raise ChatDatasetError(f"chat dataset split file is missing or altered: {record.file}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if tuple(row["record"] for row in rows) != record.record_ids:
        raise ChatDatasetError("chat dataset split rows do not match the manifest")
    if check_removals:
        from epor.alignment.records import AlignmentStore

        index = AlignmentStore(store).removal_index()
        removed = [item for item in record.record_ids if index.removed(item)]
        if removed:
            raise ChatDatasetError(
                f"{len(removed)} record(s) in this dataset were removed after it was built; "
                "rebuild the dataset so removed material never reaches training"
            )
    return manifest, rows


class ChatCorpus:
    """One split of an ``sft`` chat dataset as masked next-token pairs."""

    def __init__(
        self,
        *,
        dataset_id: str,
        manifest: ChatDatasetManifest,
        split: Split,
        ids: list[Tensor],
        masks: list[Tensor],
        record_ids: tuple[str, ...],
        max_context: int | None,
    ) -> None:
        self.tokenized_id = dataset_id
        self.manifest = manifest
        self.split: Split = split
        self.document_ids = record_ids
        self._ids = ids
        self._masks = masks
        # Prefix sums of trainable targets: _targets[d][j] = targets among pairs < j.
        self._targets = [
            torch.cat((torch.zeros(1, dtype=torch.int64), mask[1:].cumsum(0))) for mask in masks
        ]
        self.max_context = max_context
        self.pair_counts: tuple[int, ...] = tuple(len(item) - 1 for item in ids)

    @property
    def tokenizer_id(self) -> str:
        return self.manifest.tokenizer_id

    @property
    def vocab_size(self) -> int:
        return self.manifest.vocab_size

    @property
    def document_count(self) -> int:
        return len(self._ids)

    @classmethod
    def open(
        cls,
        data_root: str | Path | DataStore,
        dataset_id: str,
        split: Split,
        *,
        model_config: ModelConfig | None = None,
        tokenizer_id: str | None = None,
    ) -> ChatCorpus:
        manifest, rows = read_split_rows(data_root, dataset_id, split, kind="sft")
        if tokenizer_id is not None and manifest.tokenizer_id != tokenizer_id:
            raise ChatDatasetError(
                "chat dataset was tokenized with a different tokenizer than the model's build"
            )
        if model_config is not None:
            if manifest.vocab_size != model_config.vocab_size:
                raise ChatDatasetError(
                    f"chat dataset vocabulary {manifest.vocab_size} does not match the model "
                    f"vocab_size {model_config.vocab_size}"
                )
            special = (
                model_config.pad_token_id,
                model_config.bos_token_id,
                model_config.eos_token_id,
            )
            if special != (PAD_ID, BOS_ID, EOS_ID):
                raise ChatDatasetError(
                    "model pad/bos/eos token IDs do not match the reserved tokenizer spec"
                )
        ids: list[Tensor] = []
        masks: list[Tensor] = []
        for row in rows:
            tokens = torch.tensor(row["ids"], dtype=torch.int64)
            mask = torch.tensor([flag == "1" for flag in row["mask"]], dtype=torch.bool)
            if tokens.numel() < 2 or tokens.numel() != mask.numel():
                raise ChatDatasetError("chat dataset row is malformed")
            if int(tokens.max()) >= manifest.vocab_size or int(tokens.min()) < 0:
                raise ChatDatasetError("chat dataset row holds an out-of-vocabulary ID")
            ids.append(tokens)
            masks.append(mask)
        if not ids and split == "train":
            raise ChatDatasetError("chat dataset training split is empty")
        return cls(
            dataset_id=dataset_id,
            manifest=manifest,
            split=split,
            ids=ids,
            masks=masks,
            record_ids=tuple(row["record"] for row in rows),
            max_context=None if model_config is None else model_config.configured_max_context,
        )

    def pairs(self, document: int, start: int, length: int) -> tuple[Tensor, Tensor]:
        if length < 1 or start + length > self.pair_counts[document]:
            raise IndexError("pair range is outside the document")
        tokens = self._ids[document]
        inputs = tokens[start : start + length]
        targets = tokens[start + 1 : start + length + 1].clone()
        targets[~self._masks[document][start + 1 : start + length + 1]] = IGNORE_INDEX
        return inputs, targets

    def target_count(self, document: int, start: int, length: int) -> int:
        prefix = self._targets[document]
        return int(prefix[start + length] - prefix[start])


__all__ = [
    "ChatCorpus",
    "ChatDatasetError",
    "ChatDatasetManifest",
    "ChatDatasetSplit",
    "DatasetKind",
    "dataset_directory",
    "load_chat_manifest",
    "read_split_rows",
]
