"""Fault injection: crashes between and during checkpoints recover exactly (G3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from conftest import TokenShardFixture, make_token_shards  # noqa: E402
from kernel_support import (  # noqa: E402
    kernel_config,
    kernel_settings,
    same_weights,
    train_worker,
)

from epor.training import (  # noqa: E402
    FaultPlan,
    InjectedFault,
    latest_checkpoint,
    launch_local,
    pretrain_packed,
    read_kernel_checkpoint,
)
from epor.training.kernel import final_model_state  # noqa: E402


@pytest.fixture(scope="module")
def shards(tmp_path_factory: pytest.TempPathFactory) -> TokenShardFixture:
    return make_token_shards(tmp_path_factory.mktemp("kernel-faults"))


@pytest.mark.parametrize(
    ("point", "fault_step", "expected_latest"),
    [("step", 3, 2), ("checkpoint_write", 4, 2)],
)
def test_crash_recovery_reproduces_uninterrupted_run(
    shards: TokenShardFixture,
    tmp_path: Path,
    point: str,
    fault_step: int,
    expected_latest: int,
) -> None:
    config = kernel_config(shards.vocab_size)
    settings = kernel_settings(shards.tokenized_id)
    full = pretrain_packed(config, shards.data_root, tmp_path / "full", settings, max_steps=6)

    run = tmp_path / "crashed"
    with pytest.raises(InjectedFault):
        pretrain_packed(
            config,
            shards.data_root,
            run,
            settings,
            max_steps=6,
            fault_plan=FaultPlan(step=fault_step, point=point),  # type: ignore[arg-type]
        )
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["termination_reason"] == "error:InjectedFault"
    # The interrupted write never became "latest" and the good one still verifies.
    assert read_kernel_checkpoint(run)["step"] == expected_latest
    abandoned = list((run / "checkpoints").glob(".tmp-*"))
    assert bool(abandoned) == (point == "checkpoint_write")

    resumed = pretrain_packed(config, shards.data_root, run, settings, max_steps=6, resume_from=run)
    assert not list((run / "checkpoints").glob(".tmp-*"))
    assert resumed.losses == full.losses[expected_latest:]
    assert same_weights(final_model_state(tmp_path / "full"), final_model_state(run))


def test_killed_distributed_checkpoint_write_recovers_on_one_process(
    shards: TokenShardFixture, tmp_path: Path
) -> None:
    """A rank killed mid-checkpoint (no cleanup) leaves the previous checkpoint
    intact; a single process then resumes the 2-rank DDP run (a world-size
    change) and reproduces the uninterrupted single-process trajectory."""

    config = kernel_config(shards.vocab_size)
    settings = kernel_settings(shards.tokenized_id)
    full = pretrain_packed(config, shards.data_root, tmp_path / "full", settings, max_steps=6)

    run = tmp_path / "ddp"
    job = {
        "strategy": "ddp",
        "config": config,
        "data_root": shards.data_root,
        "output": run,
        "settings": settings,
        "max_steps": 6,
        "fault": FaultPlan(step=4, point="checkpoint_write", mode="exit", ranks=(1,)),
    }
    # Rank 1 exits (code 86); rank 0 then fails in the commit barrier.
    with pytest.raises(RuntimeError, match=r"distributed (launch failed|rank exited)"):
        launch_local(train_worker, 2, [job], timeout_seconds=90, threads_per_rank=2)
    latest = latest_checkpoint(run)
    assert latest is not None and latest.name == "step-00000002"
    assert read_kernel_checkpoint(latest)["saved_world_size"] == 2

    resumed = pretrain_packed(config, shards.data_root, run, settings, max_steps=6, resume_from=run)
    assert resumed.losses == full.losses[2:]
    assert same_weights(final_model_state(tmp_path / "full"), final_model_state(run))
