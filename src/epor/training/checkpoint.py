"""Safe-by-construction local checkpoint and atomic artifact helpers."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import torch

from epor.reference_policy import validate_reference_checkpoint

from .data import sha256_file

CHECKPOINT_SCHEMA_VERSION = "1"


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write canonical JSON and replace the destination atomically."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    """Atomically save a tensor-only trusted checkpoint at a step boundary."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    complete_payload = dict(payload)
    complete_payload["schema_version"] = CHECKPOINT_SCHEMA_VERSION
    try:
        torch.save(complete_payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load an EPOR checkpoint with PyTorch's restricted weights-only loader."""

    checkpoint_path = validate_reference_checkpoint(path)
    payload = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema version")
    required = {
        "run_id",
        "step",
        "model_config",
        "model_state",
        "optimizer_state",
        "training_settings",
        "data_state",
        "torch_rng_state",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"checkpoint is missing fields: {', '.join(missing)}")
    if not isinstance(payload["step"], int) or payload["step"] < 0:
        raise ValueError("checkpoint step is invalid")
    if not isinstance(payload["torch_rng_state"], torch.Tensor):
        raise ValueError("checkpoint CPU RNG state is invalid")
    return payload


def checkpoint_artifact(path: str | Path, *, root: str | Path) -> dict[str, Any]:
    """Describe a checkpoint with a stable relative path and checksum."""

    artifact_path = Path(path).resolve()
    root_path = Path(root).resolve()
    try:
        relative = artifact_path.relative_to(root_path)
    except ValueError:
        relative = artifact_path
    return {
        "path": relative.as_posix(),
        "sha256": sha256_file(artifact_path),
        "size_bytes": artifact_path.stat().st_size,
    }
