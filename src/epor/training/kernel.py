"""The v0.0.5 training kernel: packed token shards, AdamW, BF16, DDP/FSDP2, exact resume.

Entry points
------------
:func:`pretrain_packed`
    Train (or resume) on one split of a verified tokenized build.  Settings
    live in :class:`PackedTrainingSettings`; process topology in a
    :class:`~epor.training.distributed.DistributedContext` (default: one
    process).  Returns :class:`PackedTrainingResult`.
:func:`evaluate_packed_checkpoint`
    Single-process validation loss/perplexity (+ per-depth MTP loss) of a
    kernel checkpoint.  Returns :class:`PackedEvaluationResult`.

The v0.0.1 debug path (:func:`epor.training.pretrain` on the byte fixture)
is separate and unchanged.

One optimizer step
------------------
1. Reseed the global torch RNG from ``(model_seed, step, rank)`` so dropout
   is a pure function of the step and survives any resume.
2. Draw this rank's micro-batches from :class:`PackedBatchStream`.
3. Each micro-batch's backward objective is token-normalized over the *global*
   step: ``main_loss * target_tokens / global_target_tokens`` plus, with MTP,
   ``mtp_loss_weight / D * Σ_d mtp_loss_d * valid_d / global_valid_d``, all
   multiplied by ``world_size`` because DDP and FSDP2 average gradients.  The
   summed gradient therefore equals the gradient of the single-batch model
   loss over the whole global step, independent of world size and
   accumulation.  ``global_valid_d`` (MTP targets per depth) is gathered from
   all ranks before the first backward.
4. Non-final micro-batches skip gradient reduction (``no_sync`` /
   ``set_requires_gradient_sync(False)``).
5. Clip the global gradient norm, set the scheduled learning rate, step AdamW.
6. Gather per-micro-batch loss sums in global micro-batch order and sum them in
   that order, so the logged loss is the same number for every topology that
   computes the same per-micro-batch values.

Data mixtures
-------------
``train_mixture`` (default ``None``: the whole split) restricts the training
split to a deterministic domain-weighted subset
(:func:`~epor.training.mixture.select_mixture_documents`).  It is trajectory,
so a resume must keep it; at ``None`` it is left out of
:meth:`PackedTrainingSettings.trajectory`, so every pre-existing trajectory
hash and checkpoint stays compatible.  Validation is never re-weighted.

Precision is ``bf16`` (CPU autocast around the forward; FP32 parameters,
gradients, optimizer state, and loss) or ``fp32``.  FP8 is not implemented:
it is a cloud-only ablation in the roadmap.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import time
import uuid
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from epor.data.models import Sha256
from epor.models.config import ModelConfig, load_model_config, model_config_from_mapping
from epor.models.reference import IGNORE_INDEX, DecoderOutput, EporDecoder
from epor.reference_policy import validate_kernel_model, validate_kernel_run
from epor.tokenizer.corpus import Split

from .checkpoint import atomic_write_json
from .distributed import DistributedContext, gradient_sync, wrap_model
from .faults import FaultPlan
from .loop import (
    PROJECT_ROOT,
    TrainingCancelled,
    _append_metric,  # pyright: ignore[reportPrivateUsage]
    _write_manifest,  # pyright: ignore[reportPrivateUsage]
)
from .manifest import new_manifest, resource_usage, utc_now
from .mixture import DomainMixture, MixtureSelection, document_domains, select_from_corpus
from .optim import (
    OptimizerSettings,
    build_optimizer,
    clip_gradients,
    learning_rate_at,
    set_learning_rate,
)
from .packing import PackedBatch, PackedBatchStream, evaluation_steps
from .run_checkpoint import (
    CheckpointError,
    latest_checkpoint,
    load_kernel_checkpoint,
    read_kernel_checkpoint,
    remove_abandoned,
    save_kernel_checkpoint,
)
from .shard_data import ShardCorpus

Precision = Literal["bf16", "fp32"]

# Settings that change speed, memory, or bookkeeping but never the math of a
# step; they may differ across a resume.  Everything else is trajectory.
RUNTIME_FIELDS = frozenset(
    {
        "micro_batch_size",
        "activation_checkpointing",
        "checkpoint_every",
        "keep_checkpoints",
        "validation_split",
        "eval_max_steps",
    }
)


class PackedTrainingSettings(BaseModel):
    """Training-kernel settings; see :data:`RUNTIME_FIELDS` for what may change on resume.

    ``micro_batch_size`` is rows per micro-batch *per rank* (default: all of
    the rank's rows in one micro-batch).  ``tokens_per_batch`` is the global
    optimizer-step budget in positions and must be a multiple of ``seq_len``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    tokenized_id: Sha256
    train_split: Split = "train"
    validation_split: Split | None = "validation"
    seq_len: int = Field(default=128, ge=1)
    tokens_per_batch: int = Field(default=4_096, ge=1)
    micro_batch_size: int | None = Field(default=None, ge=1)
    data_seed: int = Field(default=23, ge=0)
    model_seed: int = Field(default=17, ge=0)
    optimizer: OptimizerSettings = Field(default_factory=OptimizerSettings)
    precision: Precision = "bf16"
    activation_checkpointing: bool = False
    checkpoint_every: int = Field(default=50, ge=1)
    keep_checkpoints: int = Field(default=3, ge=1)
    eval_max_steps: int | None = Field(default=None, ge=1)
    train_mixture: DomainMixture | None = None

    def trajectory(self) -> dict[str, Any]:
        payload = {
            key: value
            for key, value in self.model_dump(mode="json").items()
            if key not in RUNTIME_FIELDS
        }
        # Added after trajectories were hashed into checkpoints; at its
        # default it selects the original data and stays out of the hash.
        if payload["train_mixture"] is None:
            del payload["train_mixture"]
        return payload

    @property
    def trajectory_sha256(self) -> str:
        encoded = json.dumps(self.trajectory(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


class PackedEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True)

    split: Split
    main_loss: float
    perplexity: float
    mtp_losses: tuple[float, ...]
    target_tokens: int
    steps: int


class PackedTrainingResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    lineage_id: str
    output_dir: Path
    checkpoint_path: Path
    manifest_path: Path
    start_step: int
    steps_completed: int
    losses: tuple[float, ...]
    main_losses: tuple[float, ...]
    tokens_seen: int
    evaluation: PackedEvaluation | None


class PackedEvaluationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    checkpoint_path: Path
    model_id: str
    evaluation: PackedEvaluation
    report_path: Path | None


class _StepRecord(BaseModel):
    """One micro-batch's loss sums; gathered across ranks in micro-batch order."""

    main_sum: float
    target_tokens: int
    mtp_sums: tuple[float, ...]


def mtp_target_counts(batch: PackedBatch, depth: int) -> list[int]:
    """Valid MTP targets per depth, masked exactly as ``EporDecoder`` masks them."""

    counts: list[int] = []
    sequence = batch.targets.shape[1]
    for offset in range(1, depth + 1):
        length = sequence - offset
        valid = batch.targets[:, offset:] != IGNORE_INDEX
        valid &= batch.segment_ids[:, offset:] == batch.segment_ids[:, :length]
        counts.append(int(valid.sum()))
    return counts


def micro_objective(
    output: DecoderOutput,
    *,
    target_tokens: int,
    mtp_counts: Sequence[int],
    global_target_tokens: int,
    global_mtp_counts: Sequence[int],
    mtp_loss_weight: float,
    world_size: int = 1,
) -> Tensor:
    """One micro-batch's share of the global-step objective (see the module docstring).

    Summing the gradients of these objectives over every micro-batch and rank
    (with averaging across ``world_size``) gives exactly the gradient of
    ``main_loss + mtp_loss_weight * mean_d(mtp_loss_d)`` evaluated on the whole
    global step at once.
    """

    if output.main_loss is None:
        raise RuntimeError("decoder did not return a training loss")
    objective = output.main_loss * (target_tokens * world_size / global_target_tokens)
    depth_count = len(global_mtp_counts)
    for depth, (loss, count) in enumerate(zip(output.mtp_losses, mtp_counts, strict=True)):
        if count:
            share = mtp_loss_weight / depth_count * count * world_size
            objective = objective + loss * (share / global_mtp_counts[depth])
    return objective


def _step_seed(seed: int, step: int, rank: int) -> int:
    digest = hashlib.sha256(f"epor-kernel-step:{seed}:{step}:{rank}".encode()).digest()
    return int.from_bytes(digest[:8], "little") >> 1


def _autocast(precision: Precision) -> Any:
    if precision == "bf16":
        return torch.autocast("cpu", dtype=torch.bfloat16)
    return nullcontext()


def resolve_model_config(model: ModelConfig | str | Path) -> ModelConfig:
    config = model if isinstance(model, ModelConfig) else load_model_config(model)
    return validate_kernel_model(config)


def _collective(context: DistributedContext, action: Callable[[], Any]) -> Any:
    """Run ``action`` on rank 0 and re-raise its failure on every rank."""

    outcome: tuple[bool, Any] = (True, None)
    if context.is_main:
        try:
            outcome = (True, action())
        except Exception as exc:
            outcome = (False, (type(exc).__name__, str(exc)))
    ok, value = context.broadcast_object(outcome)
    if not ok:
        name, message = value
        error_type = {
            "FileExistsError": FileExistsError,
            "CheckpointError": CheckpointError,
        }.get(name, ValueError)
        raise error_type(message)
    return value


def _check_compatible(
    trainer: dict[str, Any], config: ModelConfig, settings: PackedTrainingSettings
) -> None:
    saved_config = model_config_from_mapping(trainer.get("model_config"))
    if saved_config.sha256 != config.sha256:
        raise CheckpointError(
            "resume model config differs from the checkpoint "
            f"({config.sha256[:12]} != {saved_config.sha256[:12]})"
        )
    saved = trainer.get("trajectory")
    current = settings.trajectory()
    if saved != current:
        differing = sorted(
            key
            for key in set(current) | set(saved or {})
            if (saved or {}).get(key) != current.get(key)
        )
        raise CheckpointError(
            "resume settings change the training trajectory: " + ", ".join(differing)
        )


def _evaluate(
    model: nn.Module,
    corpus: ShardCorpus,
    settings: PackedTrainingSettings,
    context: DistributedContext,
    *,
    max_steps: int | None,
    mtp_depth: int,
    should_cancel: Callable[[], bool] | None = None,
    on_batch: Callable[[int], None] | None = None,
) -> PackedEvaluation:
    """Token-weighted validation loss over one unshuffled pass of ``corpus``.

    Every rank runs every forward (FSDP2 forwards are collective); padding-only
    micro-batches are simply not counted.
    """

    was_training = model.training
    model.eval()
    records: list[tuple[float, int, list[float], list[int]]] = []
    steps = 0
    try:
        with torch.no_grad():
            for packed in evaluation_steps(
                corpus,
                seq_len=settings.seq_len,
                tokens_per_batch=settings.tokens_per_batch,
                rank=context.rank,
                world_size=context.world_size,
                micro_batch_size=settings.micro_batch_size,
                max_steps=max_steps,
            ):
                if context.any(should_cancel is not None and should_cancel()):
                    raise TrainingCancelled("evaluation cancellation requested")
                for micro in packed.micro_batches:
                    with _autocast(settings.precision):
                        output = model(
                            micro.input_ids, micro.targets, segment_ids=micro.segment_ids
                        )
                    if micro.target_tokens == 0 or output.main_loss is None:
                        continue
                    counts = mtp_target_counts(micro, mtp_depth)
                    records.append(
                        (
                            float(output.main_loss) * micro.target_tokens,
                            micro.target_tokens,
                            [
                                float(loss) * count
                                for loss, count in zip(output.mtp_losses, counts, strict=True)
                            ],
                            counts,
                        )
                    )
                steps += 1
                if on_batch is not None and context.is_main:
                    on_batch(steps)
    finally:
        model.train(was_training)
    gathered = [record for rank in context.gather_objects(records) for record in rank]
    target_tokens = sum(record[1] for record in gathered)
    if target_tokens == 0:
        raise ValueError(f"split {corpus.split!r} produced no evaluation targets")
    main_loss = sum(record[0] for record in gathered) / target_tokens
    mtp_losses = []
    for depth in range(mtp_depth):
        valid = sum(record[3][depth] for record in gathered)
        total = sum(record[2][depth] for record in gathered)
        mtp_losses.append(total / valid if valid else float("nan"))
    return PackedEvaluation(
        split=corpus.split,
        main_loss=main_loss,
        perplexity=math.exp(min(main_loss, 80.0)),
        mtp_losses=tuple(mtp_losses),
        target_tokens=target_tokens,
        steps=steps,
    )


def _prepare_output(
    destination: Path,
    *,
    trainer: dict[str, Any] | None,
    checkpoint_dir: Path | None,
) -> None:
    if trainer is None:
        try:
            destination.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"new training output must not already exist: {destination}"
            ) from exc
        return
    if not destination.exists():
        destination.mkdir(parents=True)
        return
    try:
        current = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CheckpointError("an existing resume output must hold a lineage manifest") from exc
    if current.get("lineage_id") != trainer.get("lineage_id"):
        raise CheckpointError("resume output belongs to a different run lineage")
    assert checkpoint_dir is not None
    try:
        checkpoint_dir.relative_to(destination / "checkpoints")
    except ValueError as exc:
        raise CheckpointError("resume checkpoint is not part of the output's lineage") from exc


def pretrain_packed(
    model: ModelConfig | str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    settings: PackedTrainingSettings,
    *,
    max_steps: int,
    resume_from: str | Path | None = None,
    context: DistributedContext | None = None,
    project_root: str | Path | None = None,
    should_cancel: Callable[[], bool] | None = None,
    on_step: Callable[[int, int, float], None] | None = None,
    fault_plan: FaultPlan | None = None,
    verify_data: bool = True,
) -> PackedTrainingResult:
    """Train on token shards to the absolute optimizer step ``max_steps``.

    ``model`` is a :class:`ModelConfig` or recipe path; its vocabulary must
    equal the tokenized build's.  ``resume_from`` is a checkpoint directory, a
    run directory, or its ``checkpoints`` directory (``latest.json`` is
    followed).  Resuming into an existing ``output_dir`` requires the same
    lineage; a new ``output_dir`` starts a child lineage record.  Trajectory
    settings, model config, and the data pipeline must match the checkpoint
    (:class:`CheckpointError` otherwise); world size and strategy may change.

    Every rank must call this with identical arguments.  Rank 0 alone writes
    ``manifest.json``, ``manifests/<run_id>.json``, ``metrics.jsonl`` and
    calls ``on_step(step, max_steps, loss)``.  ``should_cancel`` is polled on
    every rank before and after each step (any rank's request stops all of
    them); a cancelled run first checkpoints its last completed step, then
    raises :class:`TrainingCancelled`.  Stopping is never gated.
    """

    context = context or DistributedContext()
    validate_kernel_run(max_steps, settings.tokens_per_batch)
    destination = Path(output_dir).expanduser().resolve()
    config = resolve_model_config(model)
    corpus = ShardCorpus.open(
        Path(data_root),
        settings.tokenized_id,
        settings.train_split,
        model_config=config,
        verify=verify_data,
    )
    selection: MixtureSelection | None = None
    if settings.train_mixture is not None:
        selection = apply_mixture(Path(data_root), corpus, settings.train_mixture)
        corpus = corpus.select(selection.document_ids)
    stream = PackedBatchStream(
        corpus,
        seq_len=settings.seq_len,
        tokens_per_batch=settings.tokens_per_batch,
        seed=settings.data_seed,
        rank=context.rank,
        world_size=context.world_size,
        micro_batch_size=settings.micro_batch_size,
    )

    trainer: dict[str, Any] | None = None
    checkpoint_dir: Path | None = None
    if resume_from is not None:
        trainer = read_kernel_checkpoint(resume_from)
        checkpoint_dir = Path(trainer["checkpoint_dir"])
        _check_compatible(trainer, config, settings)
        stream.load_state_dict(trainer["data_state"])
        if max_steps <= int(trainer["step"]):
            raise ValueError("max_steps must exceed the checkpoint step when resuming")

    _collective(
        context,
        lambda: _prepare_output(destination, trainer=trainer, checkpoint_dir=checkpoint_dir),
    )
    checkpoints_root = destination / "checkpoints"
    if context.is_main:
        remove_abandoned(checkpoints_root)
    lineage_id = (
        str(trainer["lineage_id"]) if trainer is not None else f"lineage-{uuid.uuid4().hex}"
    )
    lineage_id = context.broadcast_object(lineage_id)
    run_id = context.broadcast_object(f"kernel-{uuid.uuid4().hex}")
    start_step = int(trainer["step"]) if trainer is not None else 0
    tokens_seen = int(trainer["tokens_seen"]) if trainer is not None else 0
    target_tokens_seen = int(trainer["target_tokens_seen"]) if trainer is not None else 0

    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    manifest: dict[str, Any] = {}
    if context.is_main:
        manifest = new_manifest(
            run_id=run_id,
            config_hash=config.sha256,
            data_hash=settings.tokenized_id,
            tokenizer_hash=corpus.tokenizer_id,
            model_seed=settings.model_seed,
            data_seed=settings.data_seed,
            started_at=utc_now(),
            parent_run_id=None if trainer is None else str(trainer["run_id"]),
            parent_checkpoint=None if checkpoint_dir is None else str(checkpoint_dir),
            project_root=Path(project_root).resolve() if project_root else PROJECT_ROOT,
        )
        manifest.update(
            command="kernel_train",
            lineage_id=lineage_id,
            precision=settings.precision,
            training_settings=settings.model_dump(mode="json"),
            trajectory_sha256=settings.trajectory_sha256,
            optimizer=settings.optimizer.model_dump(mode="json"),
            distributed={"strategy": context.strategy, "world_size": context.world_size},
            data_pipeline={
                **stream.settings.model_dump(mode="json"),
                "fingerprint": stream.settings.fingerprint,
                "accumulation_steps": stream.geometry.accumulation_steps,
                "micro_batch_size": stream.geometry.micro_batch_size,
            },
            tokenized_id=settings.tokenized_id,
            tokenizer_id=corpus.tokenizer_id,
            model_id=config.slug,
            model_features={
                "qk_norm": config.qk_norm,
                "mtp_depth": config.mtp_depth,
                "mtp_loss_weight": config.mtp_loss_weight,
                "tie_embeddings": config.tie_embeddings,
            },
            activation_checkpointing=settings.activation_checkpointing,
            train_mixture=None if selection is None else selection.summary(),
            start_step=start_step,
            target_steps=max_steps,
            counters={"tokens_seen": tokens_seen, "target_tokens_seen": target_tokens_seen},
            throughput=None,
            evaluation=None,
        )
        _write_manifest(destination, manifest)

    deterministic_before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    losses: list[float] = []
    main_losses: list[float] = []
    last_checkpoint = checkpoint_dir
    last_saved_step = start_step if trainer is not None else -1
    step_seconds = 0.0
    trained_tokens = 0
    evaluation: PackedEvaluation | None = None
    mtp_depth = config.mtp_depth
    world = context.world_size

    def checkpoint(step: int, wrapped: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        nonlocal last_checkpoint, last_saved_step

        def before_commit() -> None:
            if fault_plan is not None:
                fault_plan.maybe_fire("checkpoint_write", step, context.rank)

        last_checkpoint = save_kernel_checkpoint(
            checkpoints_root,
            step=step,
            model=wrapped,
            optimizer=optimizer,
            trainer_state={
                "lineage_id": lineage_id,
                "run_id": run_id,
                "tokens_seen": tokens_seen,
                "target_tokens_seen": target_tokens_seen,
                "data_state": stream.state_dict(),
                "model_config": config.model_dump(mode="json"),
                "model_config_sha256": config.sha256,
                "settings": settings.model_dump(mode="json"),
                "trajectory": settings.trajectory(),
                "trajectory_sha256": settings.trajectory_sha256,
                "strategy": context.strategy,
                "created_at": utc_now(),
                "metrics": {"loss": losses[-1] if losses else None},
            },
            context=context,
            keep_last=settings.keep_checkpoints,
            before_commit=before_commit,
        )
        last_saved_step = step
        if context.is_main:
            manifest["checkpoints"].append(
                {
                    "path": last_checkpoint.relative_to(destination).as_posix(),
                    "step": step,
                }
            )
            _write_manifest(destination, manifest)

    def cancelled() -> bool:
        return context.any(should_cancel is not None and bool(should_cancel()))

    try:
        decoder = EporDecoder(
            config,
            seed=settings.model_seed,
            device="cpu",
            dtype=torch.float32,
            activation_checkpointing=settings.activation_checkpointing,
        )
        wrapped = wrap_model(decoder, context)
        optimizer = build_optimizer(wrapped, settings.optimizer)
        if checkpoint_dir is not None:
            load_kernel_checkpoint(
                checkpoint_dir, model=wrapped, optimizer=optimizer, context=context
            )
        wrapped.train()

        for step in range(start_step + 1, max_steps + 1):
            if cancelled():
                if last_saved_step != step - 1 and step - 1 > start_step:
                    checkpoint(step - 1, wrapped, optimizer)
                raise TrainingCancelled("training cancellation requested")
            began = time.perf_counter()
            torch.manual_seed(_step_seed(settings.model_seed, step, context.rank))
            packed = stream.next_step()
            global_targets = packed.global_target_tokens
            local_counts = [mtp_target_counts(micro, mtp_depth) for micro in packed.micro_batches]
            mtp_totals = [0] * mtp_depth
            if mtp_depth:
                for rank_counts in context.gather_objects(local_counts):
                    for counts in rank_counts:
                        for depth, count in enumerate(counts):
                            mtp_totals[depth] += count
            set_learning_rate(optimizer, learning_rate_at(step, settings.optimizer))
            optimizer.zero_grad(set_to_none=True)
            records: list[_StepRecord] = []
            last_index = len(packed.micro_batches) - 1
            for index, micro in enumerate(packed.micro_batches):
                if micro.target_tokens == 0:
                    raise RuntimeError("training micro-batches must contain targets")
                with gradient_sync(wrapped, index == last_index):
                    with _autocast(settings.precision):
                        output = wrapped(
                            micro.input_ids, micro.targets, segment_ids=micro.segment_ids
                        )
                    main_loss: Tensor = output.main_loss
                    micro_objective(
                        output,
                        target_tokens=micro.target_tokens,
                        mtp_counts=local_counts[index],
                        global_target_tokens=global_targets,
                        global_mtp_counts=mtp_totals,
                        mtp_loss_weight=config.mtp_loss_weight,
                        world_size=world,
                    ).backward()
                records.append(
                    _StepRecord(
                        main_sum=float(main_loss.detach()) * micro.target_tokens,
                        target_tokens=micro.target_tokens,
                        mtp_sums=tuple(
                            float(loss.detach()) * count
                            for loss, count in zip(
                                output.mtp_losses, local_counts[index], strict=True
                            )
                        ),
                    )
                )
            grad_norm = clip_gradients(wrapped.parameters(), settings.optimizer.gradient_clip)
            optimizer.step()
            step_seconds += time.perf_counter() - began
            trained_tokens += packed.global_tokens

            ordered = [record for rank in context.gather_objects(records) for record in rank]
            main_value = 0.0
            for record in ordered:
                main_value += record.main_sum
            main_value /= global_targets
            mtp_values: list[float] = []
            for depth in range(mtp_depth):
                total = 0.0
                for record in ordered:
                    total += record.mtp_sums[depth]
                mtp_values.append(total / mtp_totals[depth] if mtp_totals[depth] else 0.0)
            loss_value = main_value
            if mtp_values:
                loss_value = main_value + config.mtp_loss_weight * sum(mtp_values) / mtp_depth
            losses.append(loss_value)
            main_losses.append(main_value)
            tokens_seen += packed.global_tokens
            target_tokens_seen += global_targets

            if context.is_main:
                metric = {
                    "run_id": run_id,
                    "step": step,
                    "training_loss": loss_value,
                    "main_loss": main_value,
                    "mtp_losses": mtp_values,
                    "learning_rate": learning_rate_at(step, settings.optimizer),
                    "grad_norm": grad_norm,
                    "tokens_seen": tokens_seen,
                    "target_tokens_seen": target_tokens_seen,
                    "padding_fraction": packed.padding_fraction,
                    "epoch": stream.epoch,
                    "timestamp": utc_now(),
                }
                _append_metric(destination, metric)
                manifest["metrics"] = metric
                manifest["counters"] = {
                    "tokens_seen": tokens_seen,
                    "target_tokens_seen": target_tokens_seen,
                }
            if fault_plan is not None:
                fault_plan.maybe_fire("step", step, context.rank)
            if step % settings.checkpoint_every == 0 or step == max_steps:
                checkpoint(step, wrapped, optimizer)
            if on_step is not None and context.is_main:
                on_step(step, max_steps, loss_value)
            if step != max_steps and cancelled():
                if last_saved_step != step:
                    checkpoint(step, wrapped, optimizer)
                raise TrainingCancelled("training cancellation requested")

        if settings.validation_split is not None:
            validation = ShardCorpus.open(
                Path(data_root),
                settings.tokenized_id,
                settings.validation_split,
                model_config=config,
                verify=False,
            )
            if validation.document_count:
                evaluation = _evaluate(
                    wrapped,
                    validation,
                    settings,
                    context,
                    max_steps=settings.eval_max_steps,
                    mtp_depth=mtp_depth,
                    should_cancel=should_cancel,
                )
        if last_checkpoint is None:
            raise RuntimeError("training completed without a checkpoint")
        manifest_path = destination / "manifest.json"
        if context.is_main:
            manifest.update(
                status="succeeded",
                finished_at=utc_now(),
                actual_resource_use=resource_usage(started_wall, started_cpu),
                termination_reason="completed",
                steps_completed=max_steps,
                throughput=_throughput(trained_tokens, step_seconds, max_steps - start_step),
                evaluation=None if evaluation is None else evaluation.model_dump(mode="json"),
            )
            manifest_path = _write_manifest(destination, manifest, finalize=True)
        manifest_path = Path(context.broadcast_object(str(manifest_path)))
    except BaseException as exc:
        if context.is_main and manifest:
            cancelled_run = (
                isinstance(exc, TrainingCancelled) or exc.__class__.__name__ == "JobCancelled"
            )
            manifest.update(
                status="cancelled" if cancelled_run else "failed",
                finished_at=utc_now(),
                actual_resource_use=resource_usage(started_wall, started_cpu),
                termination_reason=(
                    "cancelled" if cancelled_run else f"error:{type(exc).__name__}"
                ),
                steps_completed=last_saved_step if last_saved_step >= 0 else start_step,
                throughput=_throughput(trained_tokens, step_seconds, len(losses)),
            )
            # Never let a manifest write failure mask the original error.
            with contextlib.suppress(OSError):
                _write_manifest(destination, manifest, finalize=True)
        raise
    finally:
        torch.use_deterministic_algorithms(deterministic_before)

    return PackedTrainingResult(
        run_id=run_id,
        lineage_id=lineage_id,
        output_dir=destination,
        checkpoint_path=last_checkpoint,
        manifest_path=manifest_path,
        start_step=start_step,
        steps_completed=max_steps,
        losses=tuple(losses),
        main_losses=tuple(main_losses),
        tokens_seen=tokens_seen,
        evaluation=evaluation,
    )


def apply_mixture(data_root: Path, corpus: ShardCorpus, mixture: DomainMixture) -> MixtureSelection:
    """The deterministic training subset ``mixture`` selects from ``corpus``."""

    manifest = corpus.manifest
    domains = document_domains(data_root, manifest.dataset_id, manifest.build_id)
    return select_from_corpus(corpus, domains, mixture)


def _throughput(tokens: int, seconds: float, steps: int) -> dict[str, Any]:
    """Measured local step throughput (forward, backward, optimizer, data)."""

    return {
        "steps": steps,
        "step_seconds": seconds,
        "tokens_per_second": tokens / seconds if seconds > 0 else None,
        "measured": "wall-clock over optimizer steps, excluding checkpoint and evaluation",
    }


def evaluate_packed_checkpoint(
    checkpoint: str | Path,
    data_root: str | Path,
    *,
    split: Split = "validation",
    max_steps: int | None = None,
    report_dir: str | Path | None = None,
    should_cancel: Callable[[], bool] | None = None,
    on_batch: Callable[[int], None] | None = None,
    document_ids: Sequence[str] | None = None,
) -> PackedEvaluationResult:
    """Evaluate a kernel checkpoint on one split in a single process.

    Uses the checkpoint's own model config and data geometry; reports main
    loss, perplexity (from main loss only), and per-depth MTP losses.  With
    ``report_dir`` a JSON report is written there atomically.
    ``document_ids`` restricts the pass to those documents of the split (for
    example one domain's validation documents).
    """

    trainer = read_kernel_checkpoint(checkpoint)
    directory = Path(trainer["checkpoint_dir"])
    config = validate_kernel_model(model_config_from_mapping(trainer["model_config"]))
    settings = PackedTrainingSettings.model_validate(trainer["settings"])
    corpus = ShardCorpus.open(Path(data_root), settings.tokenized_id, split, model_config=config)
    if document_ids is not None:
        corpus = corpus.select(document_ids)
    context = DistributedContext()
    model = EporDecoder(config, seed=0, device="cpu", dtype=torch.float32, initialize=False)
    load_kernel_checkpoint(
        directory, model=model, optimizer=None, context=context, restore_rng=False
    )
    evaluation = _evaluate(
        model,
        corpus,
        settings.model_copy(update={"micro_batch_size": None}),
        context,
        max_steps=max_steps,
        mtp_depth=config.mtp_depth,
        should_cancel=should_cancel,
        on_batch=on_batch,
    )
    report_path: Path | None = None
    if report_dir is not None:
        report_path = Path(report_dir).resolve() / f"{directory.name}-{uuid.uuid4().hex}.json"
        atomic_write_json(
            report_path,
            {
                "schema_version": "1",
                "command": "kernel_eval",
                "checkpoint": str(directory),
                "checkpoint_step": trainer["step"],
                "model_id": config.slug,
                "tokenized_id": settings.tokenized_id,
                **evaluation.model_dump(mode="json"),
                "created_at": utc_now(),
            },
        )
    return PackedEvaluationResult(
        checkpoint_path=directory,
        model_id=config.slug,
        evaluation=evaluation,
        report_path=report_path,
    )


def final_model_state(run_dir: str | Path) -> dict[str, Tensor]:
    """Full (unsharded) model weights of a run's latest checkpoint, for comparisons."""

    directory = latest_checkpoint(run_dir)
    if directory is None:
        raise CheckpointError(f"run has no complete checkpoint: {run_dir}")
    trainer = read_kernel_checkpoint(directory)
    config = model_config_from_mapping(trainer["model_config"])
    model = EporDecoder(config, seed=0, device="cpu", dtype=torch.float32, initialize=False)
    load_kernel_checkpoint(
        directory, model=model, optimizer=None, context=DistributedContext(), restore_rng=False
    )
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


__all__ = [
    "RUNTIME_FIELDS",
    "DomainMixture",
    "PackedEvaluation",
    "PackedEvaluationResult",
    "PackedTrainingResult",
    "PackedTrainingSettings",
    "apply_mixture",
    "evaluate_packed_checkpoint",
    "final_model_state",
    "micro_objective",
    "mtp_target_counts",
    "pretrain_packed",
    "resolve_model_config",
]
