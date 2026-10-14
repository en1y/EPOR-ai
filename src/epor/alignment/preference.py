"""Direct preference optimization (DPO) against a frozen reference, as a baseline.

**[disclosed]** DPO (Rafailov et al., 2023; used by DeepSeek LLM) fits a
policy to pairwise preferences without a separate reward model: for a prompt
``x`` with chosen ``y+`` and rejected ``y-``,

``loss = -log sigmoid(beta * ((log pi(y+|x) - log ref(y+|x)) - (log pi(y-|x) - log ref(y-|x))))``

where ``pi`` is the trained policy and ``ref`` a frozen copy of the starting
(SFT) checkpoint.  **[EPOR-adaptation]** It is the v0.0.7 *baseline* against
SFT; scope-limited GRPO is deferred to v0.5.0.  Log-probabilities are summed
over the reply tokens and its ``<|end_turn|>``; the prompt is never a target.

Mechanics, chosen to match the training kernel's guarantees:

* The policy starts from, and the reference is, a pinned kernel checkpoint
  (SHA-256 of ``checkpoint.json``).  Both must share one model config and one
  tokenizer with the preference dataset.  The reference is read-only.
* One process, FP32 or BF16 autocast, the kernel's AdamW groups, warmup/cosine
  schedule, and global-norm clipping.  Pairs are visited in a per-epoch
  permutation from a dedicated generator seeded by ``(seed, epoch)``, so the
  data cursor ``(epoch, position)`` plus the checkpointed weights and
  optimizer make resume exact.
* Checkpoints are ordinary kernel checkpoints (``trainer.json`` carries
  ``command: alignment_dpo`` and the parent's training settings), so the
  evaluation harness, the G10 suite, and the monitors load them unchanged.
* Every step records the DPO loss, implicit reward margin and accuracy, and
  ``policy_logratio`` — the mean per-token ``log pi - log ref`` on chosen
  replies, a sample estimate of KL(pi || ref) on the data.  When a monitor
  threshold (``configs/alignment/monitors.yaml``, labelled ``hypothesis``)
  is crossed the run alerts, or checkpoints and stops as
  :class:`MonitorAbortError`.  Cancellation checkpoints the last completed
  step and is never gated.
"""

from __future__ import annotations

import contextlib
import hashlib
import time
import uuid
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from epor.data.models import Sha256
from epor.data.store import canonical_json_bytes, sha256_bytes
from epor.models.config import model_config_from_mapping
from epor.models.reference import EporDecoder
from epor.reference_policy import validate_kernel_model
from epor.tokenizer.spec import PAD_ID
from epor.training.chat_data import ChatDatasetError, read_split_rows
from epor.training.distributed import DistributedContext
from epor.training.kernel import (
    CheckpointRef,
    PackedTrainingSettings,
    checkpoint_tokenizer_id,
    verified_checkpoint,
)
from epor.training.loop import (
    PROJECT_ROOT,
    TrainingCancelled,
    _append_metric,  # pyright: ignore[reportPrivateUsage]
    _write_manifest,  # pyright: ignore[reportPrivateUsage]
)
from epor.training.manifest import new_manifest, resource_usage, utc_now
from epor.training.optim import (
    OptimizerSettings,
    build_optimizer,
    clip_gradients,
    learning_rate_at,
    set_learning_rate,
)
from epor.training.run_checkpoint import (
    CheckpointError,
    load_kernel_checkpoint,
    read_kernel_checkpoint,
    remove_abandoned,
    save_kernel_checkpoint,
)

COMMAND = "alignment_dpo"
RUNTIME_FIELDS = frozenset({"checkpoint_every", "keep_checkpoints"})


class MonitorAbortError(RuntimeError):
    """A reward-overoptimization monitor crossed its abort threshold."""


class DpoMonitor(BaseModel):
    """Per-step guard on ``policy_logratio`` (thresholds are ``hypothesis``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    logratio_alert: float = Field(default=0.5, gt=0.0)
    logratio_abort: float = Field(default=2.0, gt=0.0)


class DpoSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    dataset_id: Sha256
    policy_init: CheckpointRef
    reference: CheckpointRef
    beta: float = Field(default=0.1, gt=0.0, le=10.0)
    pairs_per_step: int = Field(default=4, ge=1, le=64)
    seed: int = Field(default=29, ge=0)
    precision: Literal["fp32", "bf16"] = "fp32"
    optimizer: OptimizerSettings = Field(
        default_factory=lambda: OptimizerSettings(learning_rate=1e-4, weight_decay=0.0)
    )
    monitor: DpoMonitor = Field(default_factory=DpoMonitor)
    checkpoint_every: int = Field(default=25, ge=1)
    keep_checkpoints: int = Field(default=2, ge=1)

    def trajectory(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.model_dump(mode="json").items()
            if key not in RUNTIME_FIELDS
        }

    @property
    def trajectory_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.trajectory()))


class PreferenceEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True)

    split: str
    pairs: int
    loss: float
    accuracy: float
    reward_margin: float
    policy_logratio: float


class PreferenceTrainingResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    lineage_id: str
    output_dir: Path
    checkpoint_path: Path
    start_step: int
    steps_completed: int
    losses: tuple[float, ...]
    alerts: tuple[str, ...]
    evaluation: PreferenceEvaluation | None


Pair = tuple[Tensor, Tensor, Tensor]


def _pairs(rows: list[dict[str, Any]]) -> list[Pair]:
    return [
        (
            torch.tensor(row["prompt"], dtype=torch.long),
            torch.tensor(row["chosen"], dtype=torch.long),
            torch.tensor(row["rejected"], dtype=torch.long),
        )
        for row in rows
    ]


def _epoch_order(seed: int, epoch: int, count: int) -> list[int]:
    digest = hashlib.sha256(f"epor-dpo-order:{seed}:{epoch}".encode()).digest()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int.from_bytes(digest[:8], "little") >> 1)
    return torch.randperm(count, generator=generator).tolist()


def _step_seed(seed: int, step: int) -> int:
    digest = hashlib.sha256(f"epor-dpo-step:{seed}:{step}".encode()).digest()
    return int.from_bytes(digest[:8], "little") >> 1


def sequence_logprobs(model: Any, pairs: list[Pair], *, max_length: int) -> tuple[Tensor, Tensor]:
    """Summed reply log-probabilities ``[n, 2]`` (chosen, rejected) and reply lengths."""

    sequences: list[tuple[Tensor, int]] = []
    for prompt, chosen, rejected in pairs:
        for reply in (chosen, rejected):
            if reply.numel() + 1 > max_length:
                raise ValueError("a preference reply does not fit in the model's context")
            ids = torch.cat((prompt, reply))[-(max_length + 1) :]
            sequences.append((ids, reply.numel()))
    width = max(ids.numel() - 1 for ids, _ in sequences)
    inputs = torch.full((len(sequences), width), PAD_ID, dtype=torch.long)
    targets = torch.zeros((len(sequences), width), dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.bool)
    for row, (ids, reply) in enumerate(sequences):
        length = ids.numel() - 1
        inputs[row, :length] = ids[:-1]
        targets[row, :length] = ids[1:]
        mask[row, length - reply : length] = True
    logprobs = F.log_softmax(model(inputs).logits.float(), dim=-1)
    picked = logprobs.gather(2, targets[..., None])[..., 0] * mask
    lengths = torch.tensor([reply for _, reply in sequences], dtype=torch.float32)
    return picked.sum(1).view(-1, 2), lengths.view(-1, 2)


def dpo_terms(policy: Tensor, reference: Tensor, lengths: Tensor, beta: float) -> dict[str, Tensor]:
    """DPO loss and diagnostics from ``[n, 2]`` summed log-probs (chosen, rejected)."""

    ratios = policy - reference
    margin = beta * (ratios[:, 0] - ratios[:, 1])
    return {
        "loss": -F.logsigmoid(margin).mean(),
        "reward_margin": margin.detach().mean(),
        "accuracy": (margin.detach() > 0).float().mean(),
        "chosen_reward": beta * ratios[:, 0].detach().mean(),
        "rejected_reward": beta * ratios[:, 1].detach().mean(),
        "policy_logratio": (ratios[:, 0].detach() / lengths[:, 0]).mean(),
    }


def _autocast(precision: str) -> Any:
    if precision == "bf16":
        return torch.autocast("cpu", dtype=torch.bfloat16)
    return nullcontext()


def _load(directory: Path, config: Any) -> EporDecoder:
    model = EporDecoder(config, seed=0, device="cpu", dtype=torch.float32, initialize=False)
    load_kernel_checkpoint(
        directory, model=model, optimizer=None, context=DistributedContext(), restore_rng=False
    )
    return model


@torch.no_grad()
def evaluate_preferences(
    policy: Any, reference: Any, pairs: list[Pair], *, beta: float, split: str, batch: int = 8
) -> PreferenceEvaluation:
    was_training = policy.training
    policy.eval()
    terms: list[dict[str, float]] = []
    try:
        for start in range(0, len(pairs), batch):
            chunk = pairs[start : start + batch]
            context = policy.config.configured_max_context
            pol, lengths = sequence_logprobs(policy, chunk, max_length=context)
            ref, _ = sequence_logprobs(reference, chunk, max_length=context)
            values = dpo_terms(pol, ref, lengths, beta)
            terms.append({key: float(value) * len(chunk) for key, value in values.items()})
    finally:
        policy.train(was_training)
    total = len(pairs)
    return PreferenceEvaluation(
        split=split,
        pairs=total,
        loss=sum(item["loss"] for item in terms) / total,
        accuracy=sum(item["accuracy"] for item in terms) / total,
        reward_margin=sum(item["reward_margin"] for item in terms) / total,
        policy_logratio=sum(item["policy_logratio"] for item in terms) / total,
    )


def train_preference(
    data_root: str | Path,
    output_dir: str | Path,
    settings: DpoSettings,
    *,
    max_steps: int,
    resume_from: str | Path | None = None,
    project_root: str | Path | None = None,
    should_cancel: Callable[[], bool] | None = None,
    on_step: Callable[[int, int, float], None] | None = None,
) -> PreferenceTrainingResult:
    """Run DPO to the absolute optimizer step ``max_steps`` (see the module docstring)."""

    if not 1 <= max_steps <= 100_000:
        raise ValueError("max_steps must be between 1 and 100,000")
    root = Path(data_root)
    destination = Path(output_dir).expanduser().resolve()
    init_dir, init_trainer = verified_checkpoint(settings.policy_init)
    reference_dir, reference_trainer = verified_checkpoint(settings.reference)
    config = validate_kernel_model(model_config_from_mapping(init_trainer["model_config"]))
    if model_config_from_mapping(reference_trainer["model_config"]).sha256 != config.sha256:
        raise CheckpointError("the reference model config differs from the policy's")
    tokenizer_id = checkpoint_tokenizer_id(root, init_trainer)
    if checkpoint_tokenizer_id(root, reference_trainer) != tokenizer_id:
        raise CheckpointError("the reference checkpoint uses a different tokenizer")
    manifest_data, rows = read_split_rows(root, settings.dataset_id, "train", kind="preference")
    if manifest_data.tokenizer_id != tokenizer_id:
        raise ChatDatasetError("the preference dataset was tokenized with a different tokenizer")
    pairs = _pairs(rows)
    if not pairs:
        raise ChatDatasetError("the preference dataset's training split is empty")
    validation_pairs: list[Pair] = []
    if "validation" in manifest_data.splits:
        _, validation_rows = read_split_rows(
            root, settings.dataset_id, "validation", kind="preference"
        )
        validation_pairs = _pairs(validation_rows)

    trainer: dict[str, Any] | None = None
    checkpoint_dir: Path | None = None
    epoch, position, start_step = 0, 0, 0
    if resume_from is not None:
        trainer = read_kernel_checkpoint(resume_from)
        if trainer.get("command") != COMMAND:
            raise CheckpointError("resume needs a preference (DPO) checkpoint")
        if trainer.get("trajectory") != settings.trajectory():
            raise CheckpointError("resume settings change the preference-training trajectory")
        checkpoint_dir = Path(trainer["checkpoint_dir"])
        epoch, position = (
            int(trainer["data_state"]["epoch"]),
            int(trainer["data_state"]["position"]),
        )
        start_step = int(trainer["step"])
        if max_steps <= start_step:
            raise ValueError("max_steps must exceed the checkpoint step when resuming")
        existing = destination / "manifest.json"
        if existing.is_file():
            import json

            current = json.loads(existing.read_text(encoding="utf-8"))
            if current.get("lineage_id") != trainer.get("lineage_id"):
                raise CheckpointError("resume output belongs to a different run lineage")
        destination.mkdir(parents=True, exist_ok=True)
    else:
        try:
            destination.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"new training output must not already exist: {destination}"
            ) from exc
    checkpoints_root = destination / "checkpoints"
    remove_abandoned(checkpoints_root)
    lineage_id = str(trainer["lineage_id"]) if trainer else f"lineage-{uuid.uuid4().hex}"
    run_id = f"dpo-{uuid.uuid4().hex}"
    parent_settings = PackedTrainingSettings.model_validate(init_trainer["settings"])
    manifest = new_manifest(
        run_id=run_id,
        config_hash=config.sha256,
        data_hash=settings.dataset_id,
        tokenizer_hash=tokenizer_id,
        model_seed=settings.seed,
        data_seed=settings.seed,
        started_at=utc_now(),
        parent_run_id=None if trainer is None else str(trainer["run_id"]),
        parent_checkpoint=str(checkpoint_dir or init_dir),
        project_root=Path(project_root).resolve() if project_root else PROJECT_ROOT,
    )
    manifest.update(
        command=COMMAND,
        lineage_id=lineage_id,
        precision=settings.precision,
        preference_settings=settings.model_dump(mode="json"),
        trajectory_sha256=settings.trajectory_sha256,
        optimizer=settings.optimizer.model_dump(mode="json"),
        dataset_id=settings.dataset_id,
        tokenized_id=parent_settings.tokenized_id,
        tokenizer_id=tokenizer_id,
        model_id=config.slug,
        start_step=start_step,
        target_steps=max_steps,
        alerts=[],
        evaluation=None,
        claim=(
            "proxy-scale preference optimization on a local fixture model; not evidence of "
            "alignment, helpfulness, or safety"
        ),
    )
    _write_manifest(destination, manifest)

    policy = EporDecoder(config, seed=0, device="cpu", dtype=torch.float32, initialize=False)
    optimizer = build_optimizer(policy, settings.optimizer)
    if checkpoint_dir is not None:
        load_kernel_checkpoint(
            checkpoint_dir, model=policy, optimizer=optimizer, context=DistributedContext()
        )
    else:
        load_kernel_checkpoint(
            init_dir,
            model=policy,
            optimizer=None,
            context=DistributedContext(),
            restore_rng=False,
        )
    reference = _load(reference_dir, config)
    reference.eval().requires_grad_(False)
    policy.train()

    losses: list[float] = []
    alerts: list[str] = list(trainer.get("alerts", [])) if trainer else []
    last_checkpoint = checkpoint_dir
    last_saved = start_step if trainer else -1
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    order = _epoch_order(settings.seed, epoch, len(pairs))

    def checkpoint(step: int) -> None:
        nonlocal last_checkpoint, last_saved
        last_checkpoint = save_kernel_checkpoint(
            checkpoints_root,
            step=step,
            model=policy,
            optimizer=optimizer,
            trainer_state={
                "command": COMMAND,
                "lineage_id": lineage_id,
                "run_id": run_id,
                "data_state": {"epoch": epoch, "position": position},
                "model_config": config.model_dump(mode="json"),
                "model_config_sha256": config.sha256,
                # The parent's settings keep evaluation loaders working; the
                # preference trajectory lives beside them.
                "settings": parent_settings.model_dump(mode="json"),
                "preference": settings.model_dump(mode="json"),
                "trajectory": settings.trajectory(),
                "alerts": alerts,
                "created_at": utc_now(),
                "metrics": {"loss": losses[-1] if losses else None},
            },
            context=DistributedContext(),
            keep_last=settings.keep_checkpoints,
        )
        last_saved = step
        manifest.setdefault("checkpoints", []).append(
            {"path": last_checkpoint.relative_to(destination).as_posix(), "step": step}
        )
        _write_manifest(destination, manifest)

    def cancelled() -> bool:
        return should_cancel is not None and bool(should_cancel())

    deterministic_before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    evaluation: PreferenceEvaluation | None = None
    try:
        for step in range(start_step + 1, max_steps + 1):
            if cancelled():
                if last_saved != step - 1 and step - 1 > start_step:
                    checkpoint(step - 1)
                raise TrainingCancelled("preference training cancellation requested")
            torch.manual_seed(_step_seed(settings.seed, step))
            chunk: list[Pair] = []
            while len(chunk) < settings.pairs_per_step:
                chunk.append(pairs[order[position]])
                position += 1
                if position == len(pairs):
                    epoch, position = epoch + 1, 0
                    order = _epoch_order(settings.seed, epoch, len(pairs))
            set_learning_rate(optimizer, learning_rate_at(step, settings.optimizer))
            optimizer.zero_grad(set_to_none=True)
            context_length = config.configured_max_context
            with _autocast(settings.precision):
                pol, lengths = sequence_logprobs(policy, chunk, max_length=context_length)
            with torch.no_grad():
                ref, _ = sequence_logprobs(reference, chunk, max_length=context_length)
            terms = dpo_terms(pol, ref, lengths, settings.beta)
            terms["loss"].backward()
            grad_norm = clip_gradients(policy.parameters(), settings.optimizer.gradient_clip)
            optimizer.step()
            loss = float(terms["loss"].detach())
            losses.append(loss)
            logratio = float(terms["policy_logratio"])
            metric = {
                "run_id": run_id,
                "step": step,
                "training_loss": loss,
                "reward_margin": float(terms["reward_margin"]),
                "preference_accuracy": float(terms["accuracy"]),
                "chosen_reward": float(terms["chosen_reward"]),
                "rejected_reward": float(terms["rejected_reward"]),
                "policy_logratio": logratio,
                "learning_rate": learning_rate_at(step, settings.optimizer),
                "grad_norm": grad_norm,
                "epoch": epoch,
                "timestamp": utc_now(),
            }
            _append_metric(destination, metric)
            manifest["metrics"] = metric
            if logratio > settings.monitor.logratio_alert:
                alerts.append(f"step {step}: policy_logratio {logratio:.4f} above alert threshold")
                manifest["alerts"] = alerts
            if on_step is not None:
                on_step(step, max_steps, loss)
            if logratio > settings.monitor.logratio_abort:
                checkpoint(step)
                raise MonitorAbortError(
                    f"step {step}: policy_logratio {logratio:.4f} crossed the abort threshold "
                    f"{settings.monitor.logratio_abort} (hypothesis); stopped at a checkpoint"
                )
            if step % settings.checkpoint_every == 0 or step == max_steps:
                checkpoint(step)
            if step != max_steps and cancelled():
                if last_saved != step:
                    checkpoint(step)
                raise TrainingCancelled("preference training cancellation requested")
        if validation_pairs:
            evaluation = evaluate_preferences(
                policy, reference, validation_pairs, beta=settings.beta, split="validation"
            )
        assert last_checkpoint is not None
        manifest.update(
            status="succeeded",
            finished_at=utc_now(),
            actual_resource_use=resource_usage(started_wall, started_cpu),
            termination_reason="completed",
            steps_completed=max_steps,
            evaluation=None if evaluation is None else evaluation.model_dump(mode="json"),
        )
        _write_manifest(destination, manifest, finalize=True)
    except BaseException as exc:
        aborted = isinstance(exc, MonitorAbortError)
        cancelled_run = isinstance(exc, TrainingCancelled) or type(exc).__name__ == "JobCancelled"
        manifest.update(
            status="cancelled" if cancelled_run else ("aborted" if aborted else "failed"),
            finished_at=utc_now(),
            actual_resource_use=resource_usage(started_wall, started_cpu),
            termination_reason=(
                "cancelled"
                if cancelled_run
                else ("monitor-abort" if aborted else f"error:{type(exc).__name__}")
            ),
            steps_completed=last_saved if last_saved >= 0 else start_step,
        )
        with contextlib.suppress(OSError, FileExistsError):
            _write_manifest(destination, manifest, finalize=True)
        raise
    finally:
        torch.use_deterministic_algorithms(deterministic_before)
    return PreferenceTrainingResult(
        run_id=run_id,
        lineage_id=lineage_id,
        output_dir=destination,
        checkpoint_path=last_checkpoint,
        start_step=start_step,
        steps_completed=max_steps,
        losses=tuple(losses),
        alerts=tuple(alerts),
        evaluation=evaluation,
    )


__all__ = [
    "COMMAND",
    "DpoMonitor",
    "DpoSettings",
    "MonitorAbortError",
    "PreferenceEvaluation",
    "PreferenceTrainingResult",
    "dpo_terms",
    "evaluate_preferences",
    "sequence_logprobs",
    "train_preference",
]
