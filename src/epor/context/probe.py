"""Host measurements at long context: TTFT, decode throughput, peak RSS, and OOM behaviour.

Every measurement runs in a fresh ``spawn`` subprocess so that one length's
peak RSS is its own and a failure cannot take the caller down:

* the child caps its address space at its size after start-up plus
  ``memory_budget_bytes`` (``RLIMIT_AS``), so an allocation beyond the budget
  raises inside the child instead of exhausting the host;
* the parent waits at most ``timeout_seconds`` and then kills the child's
  process group, so a measurement never hangs;
* before starting, allocations that are certain (weights, the KV cache, the
  logits, and any explicit L-by-L attention mask) are added up, and a length
  whose certain bytes already exceed the budget is refused without running.

Outcomes are ``ok``, ``out-of-memory`` (an allocation failed, or the child was
killed by a signal such as the kernel's OOM killer), ``timeout``,
``refused-estimate``, ``exceeds-configured-context``, or ``error``.  Timings
and RSS are host measurements, recorded in the report's ``measured``
section; they predict nothing about other hardware.

:func:`profile_checkpoint` measures inference (prefill, i.e. time to first
token, then greedy cached decoding) for one checkpoint across lengths.
:func:`feasible_maximum` searches, per attention variant, the largest length
whose training step and inference prefill both complete within the budget.
"""

from __future__ import annotations

import multiprocessing
import os
import resource
import signal
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from epor.models.cache import Quantization, kv_cache_bytes
from epor.models.config import ModelConfig, structural_parameter_count

Mode = Literal["inference", "train"]
Status = Literal[
    "ok", "out-of-memory", "timeout", "refused-estimate", "exceeds-configured-context", "error"
]
DEFAULT_TIMEOUT_SECONDS = 300.0


def default_memory_budget() -> int:
    """Three quarters of physical memory, capped at 24 GiB."""

    pages = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return int(min(0.75 * pages, 24 * 1024**3))


def certain_components(
    config: ModelConfig,
    length: int,
    mode: Mode,
    quantize: Quantization = "none",
    *,
    rows: int = 1,
    precision: Literal["fp32", "bf16"] = "fp32",
    optimizer: bool = False,
    replicas: int = 1,
) -> dict[str, int]:
    """Allocations a run at ``length`` cannot avoid, by kind; a lower bound, not a forecast.

    The training keywords describe a kernel step on one host: ``rows`` is the
    number of sequences in flight at once across every local rank,
    ``replicas`` the number of full weight copies (one per ``ddp`` rank), and
    ``optimizer`` adds AdamW's two moments.
    """

    parameters = structural_parameter_count(config) * 4
    mask = length * length if config.local_layers_per_global else 0
    if mode == "train":
        logits = rows * length * config.vocab_size * 4
        mtp_positions = sum(max(0, length - depth) for depth in range(1, config.mtp_depth + 1))
        return {
            "weights and gradients": 2 * parameters * replicas,
            "optimizer state": 2 * parameters * replicas if optimizer else 0,
            # The logits and their gradient; under BF16 autocast the head's
            # BF16 logits sit beside the FP32 copy the loss reads.
            "logits": 2 * logits + (logits // 2 if precision == "bf16" else 0),
            # Each MTP depth keeps its own full-vocabulary log-probabilities.
            "MTP logits": rows * mtp_positions * config.vocab_size * 4,
            # Training always passes segment IDs, so every row has a boolean
            # segment mask, plus the local-window one.
            "attention masks": rows * (length * length + mask),
        }
    return {
        "weights": parameters,
        "KV cache": kv_cache_bytes(config, length, dtype_bytes=4, quantize=quantize),
        "logits": length * config.vocab_size * 4,
        "attention masks": mask,
    }


def certain_bytes(
    config: ModelConfig, length: int, mode: Mode, quantize: Quantization = "none"
) -> int:
    """Allocations a run at ``length`` cannot avoid; a lower bound, not a forecast."""

    return sum(certain_components(config, length, mode, quantize).values())


def _rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _address_space() -> int | None:
    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmSize:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _child(connection: Any, payload: dict[str, Any]) -> None:
    os.setpgrp()
    import torch

    from epor.models.cache import DecoderCache
    from epor.models.config import model_config_from_mapping
    from epor.models.reference import EporDecoder

    torch.set_num_threads(int(payload["threads"]))
    config = model_config_from_mapping(payload["config"])
    model = EporDecoder(config, seed=0)
    if payload.get("checkpoint"):
        from epor.training.distributed import DistributedContext
        from epor.training.run_checkpoint import load_kernel_checkpoint

        load_kernel_checkpoint(
            payload["checkpoint"],
            model=model,
            optimizer=None,
            context=DistributedContext(),
            restore_rng=False,
        )
    guard = False
    current = _address_space()
    if current is not None and hasattr(resource, "RLIMIT_AS"):
        limit = current + int(payload["budget"])
        try:
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            guard = True
        except (ValueError, OSError):
            guard = False
    length = int(payload["length"])
    generator = torch.Generator().manual_seed(0)
    tokens = torch.randint(256, config.vocab_size, (1, length), generator=generator)
    result: dict[str, Any] = {"memory_guard": guard}
    try:
        if payload["mode"] == "train":
            model.train()
            began = time.perf_counter()
            # The kernel always passes segment IDs, which takes attention off
            # the fused causal path; a step without them under-measures memory.
            segments = torch.zeros_like(tokens)
            output = model(tokens, torch.roll(tokens, -1, dims=1), segment_ids=segments)
            assert output.loss is not None
            output.loss.backward()
            result["train_step_seconds"] = time.perf_counter() - began
        else:
            model.eval()
            cache = DecoderCache.for_config(config, quantize=payload["quantize"])
            decode = int(payload["decode_tokens"])
            prefill = tokens[:, : length - decode] if decode else tokens
            began = time.perf_counter()
            logits = model.forward_cached(prefill, cache)
            result["ttft_seconds"] = time.perf_counter() - began
            began = time.perf_counter()
            for _ in range(decode):
                next_id = logits[:, -1:].argmax(-1)
                logits = model.forward_cached(next_id, cache)
            elapsed = time.perf_counter() - began
            result["decode_tokens_per_second"] = decode / elapsed if decode and elapsed else None
            result["cache_bytes"] = cache.nbytes()
        result["status"] = "ok"
    except MemoryError as exc:
        result.update(status="out-of-memory", message=str(exc)[:300] or "MemoryError")
    except RuntimeError as exc:
        text = str(exc)
        oom = any(word in text.lower() for word in ("alloc", "memory", "enomem"))
        result.update(status="out-of-memory" if oom else "error", message=text[:300])
    result["peak_rss_bytes"] = _rss_bytes()
    connection.send(result)
    connection.close()


def measure(
    config: ModelConfig,
    length: int,
    *,
    mode: Mode,
    checkpoint: Path | None = None,
    quantize: Quantization = "none",
    decode_tokens: int = 16,
    memory_budget_bytes: int | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    threads: int | None = None,
) -> dict[str, Any]:
    """One bounded measurement; returns ``{"status", ...}`` and never hangs."""

    budget = memory_budget_bytes or default_memory_budget()
    record: dict[str, Any] = {
        "length": length,
        "mode": mode,
        "certain_bytes": certain_bytes(config, length, mode, quantize),
        "kv_cache_bytes": kv_cache_bytes(config, length, dtype_bytes=4, quantize=quantize),
        "budget_bytes": budget,
    }
    if length > config.configured_max_context:
        return {**record, "status": "exceeds-configured-context"}
    if record["certain_bytes"] > budget:
        return {**record, "status": "refused-estimate"}
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    payload = {
        "config": config.model_dump(mode="json"),
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "length": length,
        "mode": mode,
        "quantize": quantize,
        "decode_tokens": min(decode_tokens, max(0, length - 1)),
        "budget": budget,
        "threads": threads or max(1, (os.cpu_count() or 2) - 1),
    }
    process = context.Process(target=_child, args=(send, payload), daemon=True)
    started = time.perf_counter()
    process.start()
    send.close()
    try:
        if receive.poll(timeout_seconds):
            try:
                child = receive.recv()
            except EOFError:
                child = None
        else:
            child = None
    finally:
        if process.is_alive() and process.pid is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                process.kill()
        process.join(10)
    elapsed = time.perf_counter() - started
    if child is None:
        if elapsed >= timeout_seconds:
            return {**record, "status": "timeout", "wall_seconds": elapsed}
        code = process.exitcode
        status: Status = "out-of-memory" if code is not None and code < 0 else "error"
        return {**record, "status": status, "message": f"child exited with code {code}"}
    return {**record, **child, "wall_seconds": elapsed}


def profile_checkpoint(
    checkpoint: Path,
    lengths: Sequence[int],
    *,
    quantize: Quantization = "none",
    decode_tokens: int = 16,
    memory_budget_bytes: int | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    on_length: Callable[[int], None] | None = None,
) -> tuple[ModelConfig, list[dict[str, Any]]]:
    from epor.models.config import model_config_from_mapping
    from epor.training.run_checkpoint import read_kernel_checkpoint

    trainer = read_kernel_checkpoint(checkpoint)
    config = model_config_from_mapping(trainer["model_config"])
    records = []
    for length in lengths:
        if on_length is not None:
            on_length(length)
        records.append(
            measure(
                config,
                length,
                mode="inference",
                checkpoint=Path(trainer["checkpoint_dir"]),
                quantize=quantize,
                decode_tokens=decode_tokens,
                memory_budget_bytes=memory_budget_bytes,
                timeout_seconds=timeout_seconds,
            )
        )
    return config, records


def feasible_maximum(
    variants: dict[str, ModelConfig],
    lengths: Sequence[int],
    *,
    memory_budget_bytes: int | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    on_point: Callable[[str, str, int], None] | None = None,
) -> dict[str, dict[str, Any]]:
    """Per variant and mode, measure ascending lengths until the first failure."""

    results: dict[str, dict[str, Any]] = {}
    for name, base in variants.items():
        config = ModelConfig.model_validate(
            {**base.model_dump(mode="json"), "configured_max_context": max(lengths)}
        )
        results[name] = {"config_sha256": config.sha256}
        for mode in ("train", "inference"):
            points = []
            best = 0
            for length in sorted(lengths):
                if on_point is not None:
                    on_point(name, mode, length)
                point = measure(
                    config,
                    length,
                    mode=mode,
                    memory_budget_bytes=memory_budget_bytes,
                    timeout_seconds=timeout_seconds,
                    decode_tokens=4,
                )
                points.append(point)
                if point["status"] != "ok":
                    break
                best = length
            results[name][mode] = {"max_ok_length": best, "points": points}
    return results


_MEASURED_KEYS = (
    "ttft_seconds",
    "decode_tokens_per_second",
    "train_step_seconds",
    "peak_rss_bytes",
    "wall_seconds",
    "cache_bytes",
    "memory_guard",
    "message",
)


def _split(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    content = {key: value for key, value in record.items() if key not in _MEASURED_KEYS}
    measured = {key: record[key] for key in _MEASURED_KEYS if key in record}
    return content, measured


def profile_report(
    config: ModelConfig,
    checkpoint_sha256: str | None,
    records: Sequence[dict[str, Any]],
    *,
    quantize: Quantization,
) -> Any:
    """A ``context-profile`` scaling report: statuses as content, timings as measured."""

    from epor.scaling.report import seal_scaling_report

    from .citations import RULER

    contents, measured = zip(*(_split(record) for record in records), strict=True)
    return seal_scaling_report(
        kind="context-profile",
        scope="host-measurement",
        inputs={
            "checkpoint_sha256": checkpoint_sha256,
            "config_sha256": config.sha256,
            "quantize": quantize,
            "lengths": [record["length"] for record in records],
        },
        results={"lengths": list(contents)},
        warnings=[
            "Timings and RSS describe this host only; a refusal or timeout here says nothing "
            "about other hardware."
        ],
        citations=[RULER],
        measured={"lengths": list(measured), "host": _host()},
    )


def feasibility_report(
    results: dict[str, dict[str, Any]], *, budget: int, timeout_seconds: float
) -> Any:
    from epor.scaling.report import seal_scaling_report

    content: dict[str, Any] = {}
    measured: dict[str, Any] = {}
    for name, variant in results.items():
        content[name] = {"config_sha256": variant["config_sha256"]}
        measured[name] = {}
        for mode in ("train", "inference"):
            points = [_split(point) for point in variant[mode]["points"]]
            content[name][mode] = {
                "max_ok_length": variant[mode]["max_ok_length"],
                "points": [point for point, _ in points],
            }
            measured[name][mode] = [extra for _, extra in points]
    return seal_scaling_report(
        kind="context-feasibility",
        scope="host-measurement",
        inputs={"memory_budget_bytes": budget, "timeout_seconds": timeout_seconds},
        results=content,
        warnings=[
            "The feasible maximum is the largest length whose step finished within this "
            "host's memory and time budget; it is not a trained or validated context."
        ],
        measured={"variants": measured, "host": _host()},
    )


def _host() -> dict[str, Any]:
    import platform

    return {
        "machine": platform.machine(),
        "cpus": os.cpu_count(),
        "memory_bytes": os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"),
    }


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "Mode",
    "Status",
    "certain_bytes",
    "certain_components",
    "default_memory_budget",
    "feasibility_report",
    "feasible_maximum",
    "measure",
    "profile_checkpoint",
    "profile_report",
]
