"""Single-process CPU reference pretraining, evaluation, and generation."""

from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field

from epor.models.config import (
    ModelConfig,
    dump_model_config,
    model_config_from_mapping,
)
from epor.models.reference import EporDecoder
from epor.models.tokenizer import DebugByteTokenizer
from epor.reference_policy import (
    load_reference_model_config,
    validate_reference_batch,
    validate_reference_eval_batches,
    validate_reference_generation,
    validate_reference_model,
    validate_reference_steps,
)

from .checkpoint import (
    atomic_write_json,
    checkpoint_artifact,
    load_checkpoint,
    save_checkpoint,
)
from .data import (
    DEFAULT_FIXTURE_CORPUS,
    RandomBatchStream,
    load_token_corpus,
    sequential_batches,
    sha256_file,
)
from .manifest import new_manifest, resource_usage, utc_now

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TrainingSettings(BaseModel):
    """Settings that must remain identical across an exact resume."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    batch_size: int = Field(default=1, gt=0)
    sequence_length: int = Field(default=32, gt=0)
    learning_rate: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=0.1, ge=0)
    beta1: float = Field(default=0.9, gt=0, lt=1)
    beta2: float = Field(default=0.95, gt=0, lt=1)
    gradient_clip: float = Field(default=1.0, gt=0)
    model_seed: int = Field(default=17, ge=0)
    data_seed: int = Field(default=23, ge=0)
    checkpoint_every: int = Field(default=1, gt=0)


class TrainingCancelled(RuntimeError):
    """Raised after saving state at a cooperative cancellation boundary."""


class TrainingResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    output_dir: Path
    checkpoint_path: Path
    manifest_path: Path
    steps_completed: int
    losses: tuple[float, ...]


class EvaluationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    checkpoint_path: Path
    model_id: str
    loss: float
    perplexity: float
    evaluated_tokens: int
    batches: int
    report_path: Path


class GenerationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    prompt: str
    text: str
    completion: str
    token_ids: tuple[int, ...]
    new_token_ids: tuple[int, ...]


def _setting(
    explicit: int | float | None,
    *,
    saved: TrainingSettings | None,
    name: str,
) -> int | float:
    default = TrainingSettings.model_fields[name].default
    if saved is None:
        return default if explicit is None else explicit
    saved_value = getattr(saved, name)
    if explicit is not None and explicit != saved_value:
        raise ValueError(
            f"resume setting {name}={explicit!r} does not match checkpoint value {saved_value!r}"
        )
    return saved_value


def _resolve_settings(
    *,
    saved: TrainingSettings | None,
    batch_size: int | None,
    sequence_length: int | None,
    learning_rate: float | None,
    weight_decay: float | None,
    model_seed: int | None,
    data_seed: int | None,
    checkpoint_every: int | None,
) -> TrainingSettings:
    values = {
        "batch_size": _setting(batch_size, saved=saved, name="batch_size"),
        "sequence_length": _setting(sequence_length, saved=saved, name="sequence_length"),
        "learning_rate": _setting(learning_rate, saved=saved, name="learning_rate"),
        "weight_decay": _setting(weight_decay, saved=saved, name="weight_decay"),
        "model_seed": _setting(model_seed, saved=saved, name="model_seed"),
        "data_seed": _setting(data_seed, saved=saved, name="data_seed"),
        "checkpoint_every": _setting(checkpoint_every, saved=saved, name="checkpoint_every"),
    }
    if saved is not None:
        values.update(
            beta1=saved.beta1,
            beta2=saved.beta2,
            gradient_clip=saved.gradient_clip,
        )
    return TrainingSettings.model_validate(values)


def _manifest_locations(output_dir: Path, run_id: str) -> tuple[Path, Path]:
    return output_dir / "manifest.json", output_dir / "manifests" / f"{run_id}.json"


def _write_manifest(
    output_dir: Path,
    manifest: dict[str, Any],
    *,
    finalize: bool = False,
) -> Path:
    current_path, immutable_path = _manifest_locations(output_dir, str(manifest["run_id"]))
    atomic_write_json(current_path, manifest)
    if not finalize:
        return current_path
    if immutable_path.exists():
        raise FileExistsError(f"immutable run manifest already exists: {immutable_path}")
    atomic_write_json(immutable_path, manifest)
    return immutable_path


def _prepare_output_directory(
    destination: Path,
    *,
    resume_payload: dict[str, Any] | None,
    resume_from: str | Path | None,
) -> None:
    """Reserve a new run directory or verify an exact same-lineage continuation."""

    if resume_payload is None:
        try:
            destination.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"new training output must not already exist: {destination}"
            ) from exc
        return

    if not destination.exists():
        destination.mkdir(parents=True, exist_ok=False)
        return
    if not destination.is_dir():
        raise NotADirectoryError(destination)
    current_path = destination / "manifest.json"
    try:
        current = json.loads(current_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("an existing resume output must contain a valid lineage manifest") from exc
    if not isinstance(current, dict) or current.get("run_id") != resume_payload.get("run_id"):
        raise ValueError("resume output belongs to a different run lineage")

    checkpoint = Path(resume_from).resolve() if resume_from is not None else None
    digest = sha256_file(checkpoint) if checkpoint is not None else None
    recorded = False
    for artifact in current.get("checkpoints", []):
        if not isinstance(artifact, dict) or artifact.get("sha256") != digest:
            continue
        raw_path = artifact.get("path")
        if not isinstance(raw_path, str):
            continue
        candidate = Path(raw_path)
        candidate = candidate if candidate.is_absolute() else destination / candidate
        if checkpoint is not None and candidate.resolve() == checkpoint:
            recorded = True
            break
    if not recorded:
        raise ValueError("resume checkpoint is not recorded by the output lineage manifest")


def _append_metric(output_dir: Path, metric: dict[str, Any]) -> None:
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metric, sort_keys=True, separators=(",", ":")) + "\n")


def _checkpoint_payload(
    *,
    run_id: str,
    step: int,
    config: ModelConfig,
    model: EporDecoder,
    optimizer: torch.optim.Optimizer,
    settings: TrainingSettings,
    data_stream: RandomBatchStream,
    loss: float,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "step": step,
        "created_at": utc_now(),
        "model_config": config.model_dump(mode="json"),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "training_settings": settings.model_dump(mode="json"),
        "data_state": data_stream.state_dict(),
        "torch_rng_state": torch.get_rng_state().clone(),
        "metrics": {"training_loss": loss},
    }


def _model_from_checkpoint(
    payload: dict[str, Any],
) -> tuple[ModelConfig, EporDecoder]:
    config = validate_reference_model(model_config_from_mapping(payload["model_config"]))
    model = EporDecoder(config, seed=0, device="cpu", dtype=torch.float32)
    model_state = payload["model_state"]
    if not isinstance(model_state, dict):
        raise ValueError("checkpoint model state is invalid")
    model.load_state_dict(model_state, strict=True)
    return config, model


def pretrain(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    max_steps: int = 2,
    resume_from: str | Path | None = None,
    corpus_path: str | Path = DEFAULT_FIXTURE_CORPUS,
    batch_size: int | None = None,
    sequence_length: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    model_seed: int | None = None,
    data_seed: int | None = None,
    checkpoint_every: int | None = None,
    project_root: str | Path | None = None,
    should_cancel: Callable[[], bool] | None = None,
    on_step: Callable[[int, int, float], None] | None = None,
) -> TrainingResult:
    """Run deterministic CPU FP32 next-token pretraining on the fixture corpus.

    ``max_steps`` is the absolute final optimizer step.  When ``resume_from`` is
    supplied, model, optimizer, data cursor, data-generator state, and global
    PyTorch RNG state are restored.  All trajectory-affecting settings must
    match the checkpoint; only the absolute final step may change.
    """

    validate_reference_steps(max_steps)
    destination = Path(output_dir).resolve()
    recipe_config = load_reference_model_config(config_path)
    resume_payload = load_checkpoint(resume_from) if resume_from is not None else None
    saved_settings = (
        TrainingSettings.model_validate(resume_payload["training_settings"])
        if resume_payload is not None
        else None
    )
    settings = _resolve_settings(
        saved=saved_settings,
        batch_size=batch_size,
        sequence_length=sequence_length,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        model_seed=model_seed,
        data_seed=data_seed,
        checkpoint_every=checkpoint_every,
    )
    validate_reference_batch(settings.batch_size, settings.sequence_length)
    if settings.sequence_length > recipe_config.configured_max_context:
        raise ValueError("training sequence length exceeds configured model context")
    config = ModelConfig.model_validate(
        {
            **recipe_config.model_dump(mode="json"),
            "trained_max_context": max(
                recipe_config.trained_max_context,
                settings.sequence_length,
            ),
        }
    )

    validate_reference_model(config)
    tokenizer = DebugByteTokenizer()
    corpus = load_token_corpus(corpus_path, tokenizer=tokenizer)
    data_stream = RandomBatchStream(
        corpus,
        batch_size=settings.batch_size,
        sequence_length=settings.sequence_length,
        seed=settings.data_seed,
    )

    parent_run_id: str | None = None
    start_step = 0
    model_state: dict[str, Any] = {}
    optimizer_state: dict[str, Any] = {}
    if resume_payload is not None:
        checkpoint_config = model_config_from_mapping(resume_payload["model_config"])
        if checkpoint_config.sha256 != config.sha256:
            raise ValueError("resume model config does not match config_path")
        model_state = resume_payload["model_state"]
        optimizer_state = resume_payload["optimizer_state"]
        data_state = resume_payload["data_state"]
        if not all(isinstance(value, dict) for value in (model_state, optimizer_state, data_state)):
            raise ValueError("checkpoint training state is invalid")
        # Validate corpus identity and stream settings before changing an
        # existing output lineage's mutable current-manifest pointer.
        data_stream.load_state_dict(data_state)
        start_step = resume_payload["step"]
        if max_steps <= start_step:
            raise ValueError("max_steps must exceed the checkpoint step when resuming")
        parent_run_id = str(resume_payload["run_id"])

    _prepare_output_directory(
        destination,
        resume_payload=resume_payload,
        resume_from=resume_from,
    )
    run_id = f"tiny-{uuid.uuid4().hex}"
    started_at = utc_now()
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    manifest = new_manifest(
        run_id=run_id,
        config_hash=config.sha256,
        data_hash=corpus.sha256,
        tokenizer_hash=tokenizer.sha256,
        model_seed=settings.model_seed,
        data_seed=settings.data_seed,
        started_at=started_at,
        parent_run_id=parent_run_id,
        parent_checkpoint=str(Path(resume_from).resolve()) if resume_from else None,
        project_root=Path(project_root).resolve() if project_root is not None else PROJECT_ROOT,
    )
    manifest["training_settings"] = settings.model_dump(mode="json")
    manifest["recipe_config_sha256"] = recipe_config.sha256
    manifest["target_steps"] = max_steps
    _write_manifest(destination, manifest)
    dump_model_config(config, destination / "model-config.yaml")

    deterministic_before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    losses: list[float] = []
    last_checkpoint = Path(resume_from).resolve() if resume_from else None
    try:
        torch.manual_seed(settings.model_seed)
        model = EporDecoder(
            config,
            seed=settings.model_seed,
            device="cpu",
            dtype=torch.float32,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=settings.learning_rate,
            betas=(settings.beta1, settings.beta2),
            weight_decay=settings.weight_decay,
            foreach=False,
        )
        if resume_payload is not None:
            model.load_state_dict(model_state, strict=True)
            optimizer.load_state_dict(optimizer_state)
            torch.set_rng_state(resume_payload["torch_rng_state"].cpu())

        model.train()
        for step in range(start_step + 1, max_steps + 1):
            if should_cancel is not None and should_cancel():
                raise TrainingCancelled("training cancellation requested")
            inputs, targets = data_stream.next_batch()
            optimizer.zero_grad(set_to_none=True)
            output = model(inputs, targets)
            if output.loss is None:
                raise RuntimeError("reference decoder did not return training loss")
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip)
            optimizer.step()
            loss = float(output.loss.detach())
            losses.append(loss)
            metric = {
                "run_id": run_id,
                "step": step,
                "training_loss": loss,
                "timestamp": utc_now(),
            }
            _append_metric(destination, metric)
            manifest["metrics"] = metric

            if step % settings.checkpoint_every == 0 or step == max_steps:
                checkpoint_path = destination / "checkpoints" / run_id / f"step-{step:08d}.pt"
                save_checkpoint(
                    checkpoint_path,
                    _checkpoint_payload(
                        run_id=run_id,
                        step=step,
                        config=config,
                        model=model,
                        optimizer=optimizer,
                        settings=settings,
                        data_stream=data_stream,
                        loss=loss,
                    ),
                )
                last_checkpoint = checkpoint_path
                manifest["checkpoints"].append(
                    checkpoint_artifact(checkpoint_path, root=destination)
                )
                _write_manifest(destination, manifest)

            if on_step is not None:
                on_step(step, max_steps, loss)
            if should_cancel is not None and should_cancel():
                if step % settings.checkpoint_every != 0 and step != max_steps:
                    checkpoint_path = destination / "checkpoints" / run_id / f"step-{step:08d}.pt"
                    save_checkpoint(
                        checkpoint_path,
                        _checkpoint_payload(
                            run_id=run_id,
                            step=step,
                            config=config,
                            model=model,
                            optimizer=optimizer,
                            settings=settings,
                            data_stream=data_stream,
                            loss=loss,
                        ),
                    )
                    last_checkpoint = checkpoint_path
                    manifest["checkpoints"].append(
                        checkpoint_artifact(checkpoint_path, root=destination)
                    )
                raise TrainingCancelled("training cancellation requested")

        if last_checkpoint is None:
            raise RuntimeError("training completed without a checkpoint")
        manifest["status"] = "succeeded"
        manifest["finished_at"] = utc_now()
        manifest["actual_resource_use"] = resource_usage(started_wall, started_cpu)
        manifest["termination_reason"] = "completed"
        manifest["steps_completed"] = max_steps
        manifest_path = _write_manifest(destination, manifest, finalize=True)
    except BaseException as exc:
        manifest["status"] = (
            "cancelled"
            if isinstance(exc, TrainingCancelled) or exc.__class__.__name__ == "JobCancelled"
            else "failed"
        )
        manifest["finished_at"] = utc_now()
        manifest["actual_resource_use"] = resource_usage(started_wall, started_cpu)
        manifest["termination_reason"] = (
            "cancelled"
            if isinstance(exc, TrainingCancelled) or exc.__class__.__name__ == "JobCancelled"
            else f"error:{type(exc).__name__}"
        )
        _write_manifest(destination, manifest, finalize=True)
        raise
    finally:
        torch.use_deterministic_algorithms(deterministic_before)

    return TrainingResult(
        run_id=run_id,
        output_dir=destination,
        checkpoint_path=last_checkpoint,
        manifest_path=manifest_path,
        steps_completed=max_steps,
        losses=tuple(losses),
    )


@torch.no_grad()
def evaluate(
    checkpoint_path: str | Path,
    *,
    corpus_path: str | Path = DEFAULT_FIXTURE_CORPUS,
    batch_size: int | None = None,
    sequence_length: int | None = None,
    max_batches: int | None = None,
    report_dir: str | Path | None = None,
    should_cancel: Callable[[], bool] | None = None,
    on_batch: Callable[[int], None] | None = None,
) -> EvaluationResult:
    """Evaluate a reference checkpoint on deterministic contiguous windows."""

    validate_reference_eval_batches(max_batches)
    checkpoint = Path(checkpoint_path).resolve()
    payload = load_checkpoint(checkpoint)
    config, model = _model_from_checkpoint(payload)
    saved = TrainingSettings.model_validate(payload["training_settings"])
    active_batch_size = saved.batch_size if batch_size is None else batch_size
    active_sequence_length = saved.sequence_length if sequence_length is None else sequence_length
    validate_reference_batch(active_batch_size, active_sequence_length)
    if active_sequence_length > config.configured_max_context:
        raise ValueError("evaluation sequence length exceeds configured context")
    corpus = load_token_corpus(corpus_path)
    model.eval()
    weighted_loss = 0.0
    evaluated_tokens = 0
    batches = 0
    for inputs, targets in sequential_batches(
        corpus,
        batch_size=active_batch_size,
        sequence_length=active_sequence_length,
        max_batches=max_batches,
    ):
        if should_cancel is not None and should_cancel():
            raise TrainingCancelled("evaluation cancellation requested")
        output = model(inputs, targets)
        if output.loss is None:
            raise RuntimeError("reference decoder did not return evaluation loss")
        tokens = targets.numel()
        weighted_loss += float(output.loss) * tokens
        evaluated_tokens += tokens
        batches += 1
        if on_batch is not None:
            on_batch(batches)
    if evaluated_tokens == 0:
        raise ValueError("corpus produced no complete evaluation windows")
    loss = weighted_loss / evaluated_tokens
    perplexity = math.exp(min(loss, 80.0))
    if checkpoint.parent.parent.name == "checkpoints":
        run_root = checkpoint.parents[2]
    elif checkpoint.parent.name == "checkpoints":
        # Backward-compatible layout for early v0.0.1 debug checkpoints.
        run_root = checkpoint.parent.parent
    else:
        run_root = checkpoint.parent
    evaluation_root = Path(report_dir).resolve() if report_dir else run_root / "evaluations"
    report_path = evaluation_root / f"{checkpoint.stem}-{uuid.uuid4().hex}.json"
    atomic_write_json(
        report_path,
        {
            "schema_version": "1",
            "command": "tiny_eval",
            "checkpoint": str(checkpoint),
            "checkpoint_step": payload["step"],
            "model_id": config.slug,
            "loss": loss,
            "perplexity": perplexity,
            "evaluated_tokens": evaluated_tokens,
            "batches": batches,
            "created_at": utc_now(),
        },
    )
    return EvaluationResult(
        checkpoint_path=checkpoint,
        model_id=config.slug,
        loss=loss,
        perplexity=perplexity,
        evaluated_tokens=evaluated_tokens,
        batches=batches,
        report_path=report_path,
    )


@torch.no_grad()
def generate(
    checkpoint_path: str | Path,
    prompt: str,
    max_new_tokens: int = 32,
) -> GenerationResult:
    """Greedily generate text from a reference checkpoint using debug bytes."""

    validate_reference_generation(max_new_tokens)
    payload = load_checkpoint(checkpoint_path)
    config, model = _model_from_checkpoint(payload)
    tokenizer = DebugByteTokenizer()
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    if len(prompt_ids) > config.configured_max_context:
        raise ValueError("prompt exceeds configured model context")
    inputs = torch.tensor([prompt_ids], dtype=torch.int64)
    generated = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        eos_token_id=config.eos_token_id,
    )[0].tolist()
    new_ids = generated[len(prompt_ids) :]
    return GenerationResult(
        model_id=config.slug,
        prompt=prompt,
        text=tokenizer.decode(generated),
        completion=tokenizer.decode(new_ids),
        token_ids=tuple(generated),
        new_token_ids=tuple(new_ids),
    )
