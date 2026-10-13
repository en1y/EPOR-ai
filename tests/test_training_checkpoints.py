"""Checkpoint atomicity and integrity, fault plans, and the local launcher (gate G3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from kernel_support import (  # noqa: E402
    failing_worker,
    listening_addresses_worker,
    sleeping_worker,
)

from epor.training import (  # noqa: E402
    CheckpointError,
    DistributedContext,
    FaultPlan,
    InjectedFault,
    fault_plan_from_env,
    init_distributed,
    latest_checkpoint,
    launch_local,
    read_kernel_checkpoint,
)
from epor.training.run_checkpoint import (  # noqa: E402
    load_kernel_checkpoint,
    remove_abandoned,
    save_kernel_checkpoint,
)

SINGLE = DistributedContext()


def _trained(seed: int) -> tuple[Any, Any]:
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, foreach=False)
    model(torch.randn(3, 4)).sum().backward()
    optimizer.step()
    return model, optimizer


def _save(root: Path, step: int, model: Any, optimizer: Any, **kw: Any) -> Path:
    return save_kernel_checkpoint(
        root,
        step=step,
        model=model,
        optimizer=optimizer,
        trainer_state={"note": f"step {step}"},
        context=SINGLE,
        keep_last=kw.pop("keep_last", 3),
        **kw,
    )


def test_checkpoint_round_trip_restores_model_optimizer_and_rng(tmp_path: Path) -> None:
    model, optimizer = _trained(0)
    torch.manual_seed(1234)
    expected_draw = torch.rand(3)
    torch.manual_seed(1234)
    directory = _save(tmp_path / "checkpoints", 1, model, optimizer)
    assert directory.name == "step-00000001"
    trainer = read_kernel_checkpoint(tmp_path)
    assert trainer["step"] == 1 and trainer["note"] == "step 1"
    assert trainer["saved_world_size"] == 1

    restored, restored_optimizer = _trained(99)
    torch.manual_seed(0)
    load_kernel_checkpoint(directory, model=restored, optimizer=restored_optimizer, context=SINGLE)
    assert torch.equal(torch.rand(3), expected_draw)
    for left, right in zip(
        model.state_dict().values(), restored.state_dict().values(), strict=True
    ):
        assert torch.equal(left, right)
    saved_state = optimizer.state_dict()["state"]
    loaded_state = restored_optimizer.state_dict()["state"]
    for key, values in saved_state.items():
        for name, value in values.items():
            assert torch.equal(value, loaded_state[key][name])


def test_failed_commit_never_moves_latest_and_retention_keeps_last(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    model, optimizer = _trained(0)
    for step in (1, 2, 3):
        _save(root, step, model, optimizer, keep_last=2)
    assert sorted(path.name for path in root.glob("step-*")) == ["step-00000002", "step-00000003"]
    assert latest_checkpoint(tmp_path) == root / "step-00000003"

    plan = FaultPlan(step=4, point="checkpoint_write")
    with pytest.raises(InjectedFault):
        _save(
            root,
            4,
            model,
            optimizer,
            before_commit=lambda: plan.maybe_fire("checkpoint_write", 4, 0),
        )
    assert latest_checkpoint(tmp_path) == root / "step-00000003"
    assert not (root / "step-00000004").exists()
    assert remove_abandoned(root) and not list(root.glob(".tmp-*"))


def test_tampered_or_mismatched_checkpoints_are_refused(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    model, optimizer = _trained(0)
    directory = _save(root, 1, model, optimizer)
    trainer_path = directory / "trainer.json"
    trainer = json.loads(trainer_path.read_text(encoding="utf-8"))
    trainer["note"] = "edited"
    trainer_path.write_text(json.dumps(trainer), encoding="utf-8")
    with pytest.raises(CheckpointError, match="recorded hashes"):
        read_kernel_checkpoint(directory)
    pointer = root / "latest.json"
    pointer.write_text(json.dumps({"name": "../elsewhere"}), encoding="utf-8")
    with pytest.raises(CheckpointError, match="invalid directory"):
        read_kernel_checkpoint(tmp_path)
    with pytest.raises(CheckpointError, match="not a complete"):
        read_kernel_checkpoint(tmp_path / "missing")


def test_fault_plan_validation_and_environment() -> None:
    with pytest.raises(ValueError):
        FaultPlan(step=0)
    assert fault_plan_from_env({}) is None
    plan = fault_plan_from_env({"EPOR_FAULT_AT_STEP": "3", "EPOR_FAULT_POINT": "checkpoint_write"})
    assert plan == FaultPlan(step=3, point="checkpoint_write", mode="raise")
    assert plan is not None
    with pytest.raises(ValueError):
        fault_plan_from_env({"EPOR_FAULT_AT_STEP": "3", "EPOR_FAULT_MODE": "segfault"})
    plan.maybe_fire("step", 3, 0)  # a different point never fires
    with pytest.raises(InjectedFault):
        plan.maybe_fire("checkpoint_write", 3, 0)
    FaultPlan(step=3, ranks=(1,)).maybe_fire("step", 3, 0)


def test_context_and_launcher_guard_rails(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="world_size 1"):
        DistributedContext(strategy="none", rank=0, world_size=2)
    with pytest.raises(RuntimeError, match="initialized process group"):
        DistributedContext(strategy="ddp", rank=0, world_size=2)
    for name in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="env:// variables"):
        init_distributed("fsdp2")
    assert init_distributed("none") == DistributedContext()
    with pytest.raises(RuntimeError, match="ValueError: rank"):
        launch_local(failing_worker, 2, timeout_seconds=60)
    with pytest.raises(TimeoutError):
        launch_local(sleeping_worker, 1, 120.0, timeout_seconds=4)


@pytest.mark.skipif(not Path("/proc/net/tcp").is_file(), reason="needs Linux /proc sockets")
def test_local_launch_listens_on_loopback_only() -> None:
    loopback = {"0100007F", "00000000000000000000000001000000"}
    ranks = launch_local(listening_addresses_worker, 2, timeout_seconds=60)
    addresses = [address for rank in ranks for address in rank]
    # gloo pairs listen (one per rank); a TCPStore server would add a wildcard.
    assert addresses
    for address in addresses:
        host = address.rsplit(":", 1)[0]
        assert host in loopback, address
