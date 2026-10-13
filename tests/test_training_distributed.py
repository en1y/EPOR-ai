"""Single-process vs DDP vs FSDP2 agreement and cross-topology resume (gate G3).

Runs two ``gloo`` CPU ranks through :func:`epor.training.launch_local`; every
launch has a hard timeout, and all ranks are terminated on failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from conftest import TokenShardFixture, make_token_shards  # noqa: E402
from kernel_support import (  # noqa: E402
    kernel_config,
    kernel_settings,
    max_weight_difference,
    same_weights,
    train_worker,
)

from epor.training import (  # noqa: E402
    launch_local,
    pretrain_packed,
    read_kernel_checkpoint,
)
from epor.training.kernel import final_model_state  # noqa: E402

STEPS = 4


@pytest.fixture(scope="module")
def shards(tmp_path_factory: pytest.TempPathFactory) -> TokenShardFixture:
    return make_token_shards(tmp_path_factory.mktemp("kernel-distributed"))


def _job(
    strategy: str,
    config: Any,
    shards: TokenShardFixture,
    output: Path,
    settings: Any,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "config": config,
        "data_root": shards.data_root,
        "output": output,
        "settings": settings,
        "max_steps": STEPS,
        **extra,
    }


def test_single_process_ddp_and_fsdp2_agree(shards: TokenShardFixture, tmp_path: Path) -> None:
    plain = kernel_config(shards.vocab_size)
    fp32 = kernel_settings(shards.tokenized_id)
    featured = kernel_config(shards.vocab_size, qk_norm=True, mtp_depth=1)
    bf16 = kernel_settings(
        shards.tokenized_id,
        precision="bf16",
        tokens_per_batch=64,
        activation_checkpointing=True,
    )

    single = pretrain_packed(plain, shards.data_root, tmp_path / "single", fp32, max_steps=STEPS)
    single_bf16 = pretrain_packed(
        featured, shards.data_root, tmp_path / "single-bf16", bf16, max_steps=STEPS
    )

    jobs = [
        _job("ddp", plain, shards, tmp_path / "ddp", fp32),
        _job("fsdp2", plain, shards, tmp_path / "fsdp2", fp32),
        _job("fsdp2", featured, shards, tmp_path / "fsdp2-bf16", bf16),
        # Resume the FSDP2 (sharded) step-2 checkpoint under DDP (replicated).
        _job(
            "ddp",
            plain,
            shards,
            tmp_path / "fsdp2-to-ddp",
            fp32,
            resume_from=tmp_path / "fsdp2" / "checkpoints" / "step-00000002",
        ),
    ]
    ranks = launch_local(train_worker, 2, jobs, timeout_seconds=180, threads_per_rank=2)
    assert ranks[0] == ranks[1]
    ddp, fsdp2, fsdp2_bf16, fsdp2_to_ddp = ranks[0]

    reference = final_model_state(tmp_path / "single")
    # Each rank's micro-batch is one of the single process's micro-batches, so
    # the arithmetic is identical: agreement is exact, not approximate.
    assert ddp == single.losses
    assert same_weights(final_model_state(tmp_path / "ddp"), reference)
    assert fsdp2 == single.losses
    assert same_weights(final_model_state(tmp_path / "fsdp2"), reference)
    assert fsdp2_to_ddp == single.losses[2:]
    assert same_weights(final_model_state(tmp_path / "fsdp2-to-ddp"), reference)
    assert read_kernel_checkpoint(tmp_path / "fsdp2")["strategy"] == "fsdp2"

    # BF16 + QK-norm + MTP + activation checkpointing, with two accumulation
    # micro-batches per rank: the gradient sum is associated differently
    # ((g0 + g1) + (g2 + g3) vs ((g0 + g1) + g2) + g3), so agreement is
    # within a tolerance rather than bitwise.
    assert fsdp2_bf16 == pytest.approx(single_bf16.losses, rel=1e-5, abs=1e-5)
    difference = max_weight_difference(
        final_model_state(tmp_path / "fsdp2-bf16"), final_model_state(tmp_path / "single-bf16")
    )
    assert difference < 1e-5

    # A single process resumes the sharded FSDP2 checkpoint exactly.
    resumed = pretrain_packed(
        plain,
        shards.data_root,
        tmp_path / "fsdp2-to-single",
        fp32,
        max_steps=STEPS,
        resume_from=tmp_path / "fsdp2" / "checkpoints" / "step-00000002",
    )
    assert resumed.losses == single.losses[2:]
    assert same_weights(final_model_state(tmp_path / "fsdp2-to-single"), reference)
