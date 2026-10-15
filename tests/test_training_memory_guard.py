"""The kernel's memory guard: pre-flight refusal, a soft address-space cap, and exact resume.

The over-budget allocation is a fault plan that maps 1 GiB of address space
without touching it: under the cap it fails at once, and without the cap it
succeeds without costing resident memory, so the test is safe on small CI hosts.
"""

from __future__ import annotations

import json
import resource
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from conftest import TokenShardFixture, make_token_shards  # noqa: E402
from kernel_support import (  # noqa: E402
    guarded_worker,
    kernel_config,
    kernel_settings,
    same_weights,
)

from epor.context.probe import certain_components  # noqa: E402
from epor.training import (  # noqa: E402
    FaultPlan,
    launch_local,
    pretrain_packed,
    read_kernel_checkpoint,
)
from epor.training.kernel import final_model_state  # noqa: E402
from epor.training.memory import MemoryBudgetRefusedError, preflight  # noqa: E402


@pytest.fixture(scope="module")
def shards(tmp_path_factory: pytest.TempPathFactory) -> TokenShardFixture:
    return make_token_shards(tmp_path_factory.mktemp("memory-guard"))


def _soft_limit() -> int:
    return resource.getrlimit(resource.RLIMIT_AS)[0]


def test_the_estimate_names_the_factor_that_dominates() -> None:
    config = kernel_config(65_536, configured_max_context=8_192)
    settings = kernel_settings(
        "0" * 64, seq_len=8_192, tokens_per_batch=8_192, micro_batch_size=None, precision="bf16"
    )
    with pytest.raises(MemoryBudgetRefusedError) as refused:
        preflight(config, settings, world_size=1, strategy="none", budget=4 << 30)
    # One 8,192-token row: FP32 logits and their gradient, plus the BF16 copy.
    assert "logits dominate at 5.00 GiB" in str(refused.value)
    assert "65,536-entry vocabulary" in str(refused.value)

    mtp = certain_components(kernel_config(512, mtp_depth=2), 100, "train", rows=3)
    assert mtp["MTP logits"] == 3 * (99 + 98) * 512 * 4


def test_a_run_over_budget_is_refused_before_anything_is_allocated(
    shards: TokenShardFixture, tmp_path: Path
) -> None:
    config = kernel_config(shards.vocab_size)
    settings = kernel_settings(shards.tokenized_id)
    with pytest.raises(MemoryBudgetRefusedError, match="exceed its memory budget"):
        pretrain_packed(
            config,
            shards.data_root,
            tmp_path / "refused",
            settings,
            max_steps=2,
            memory_budget_bytes=4096,
        )
    assert not (tmp_path / "refused").exists()


def test_the_manifest_records_the_budget_estimate_and_cap(
    shards: TokenShardFixture, tmp_path: Path
) -> None:
    before = _soft_limit()
    config = kernel_config(shards.vocab_size)
    settings = kernel_settings(shards.tokenized_id)
    pretrain_packed(config, shards.data_root, tmp_path / "run", settings, max_steps=2)

    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    guard = manifest["memory_guard"]
    assert guard["budget_source"] == "default"
    assert 0 < guard["estimate"]["certain_bytes"] <= guard["budget_bytes"]
    # One 16-token FP32 row: the logits and their gradient.
    assert guard["estimate"]["components"]["logits"] == 2 * 16 * shards.vocab_size * 4
    cap = guard["runtime_cap"]
    assert cap["mechanism"] == "RLIMIT_AS (soft)"
    assert cap["share_bytes"] == guard["budget_bytes"]
    assert cap["soft_limit_bytes"] <= cap["address_space_at_start_bytes"] + cap["share_bytes"]
    assert _soft_limit() == before


def test_an_allocation_past_the_cap_stops_cleanly_and_resumes_exactly(
    shards: TokenShardFixture, tmp_path: Path
) -> None:
    config = kernel_config(shards.vocab_size)
    settings = kernel_settings(shards.tokenized_id)
    full = pretrain_packed(config, shards.data_root, tmp_path / "full", settings, max_steps=6)

    run = tmp_path / "capped"
    budget = 512 << 20
    job = {
        "strategy": "none",
        "config": config,
        "data_root": shards.data_root,
        "output": run,
        "settings": settings,
        "max_steps": 6,
        "budget": budget,
        "fault": FaultPlan(step=3, mode="allocate", allocate_bytes=1 << 30),
    }
    [outcome] = launch_local(guarded_worker, 1, job, timeout_seconds=180, threads_per_rank=1)
    assert outcome["error"] == "MemoryBudgetExceededError"
    [cap] = outcome["during"]
    assert cap != resource.RLIM_INFINITY
    assert outcome["after"] == outcome["before"]

    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["termination_reason"] == "out-of-memory"
    assert manifest["steps_completed"] == 2
    assert manifest["memory_guard"]["budget_bytes"] == budget
    assert manifest["memory_guard"]["budget_source"] == "operator"
    assert manifest["memory_guard"]["runtime_cap"]["soft_limit_bytes"] == cap
    assert read_kernel_checkpoint(run)["step"] == 2

    resumed = pretrain_packed(config, shards.data_root, run, settings, max_steps=6, resume_from=run)
    assert resumed.losses == full.losses[2:]
    assert same_weights(final_model_state(tmp_path / "full"), final_model_state(run))


def test_each_local_rank_caps_itself_at_its_share(
    shards: TokenShardFixture, tmp_path: Path
) -> None:
    config = kernel_config(shards.vocab_size)
    settings = kernel_settings(shards.tokenized_id)
    budget = 1 << 30
    job = {
        "strategy": "ddp",
        "config": config,
        "data_root": shards.data_root,
        "output": tmp_path / "ddp",
        "settings": settings,
        "max_steps": 2,
        "budget": budget,
    }
    ranks = launch_local(guarded_worker, 2, job, timeout_seconds=180, threads_per_rank=1)
    for outcome in ranks:
        assert outcome["error"] is None
        [limit] = outcome["during"]
        assert limit != resource.RLIM_INFINITY
        assert outcome["after"] == outcome["before"]

    guard = json.loads((tmp_path / "ddp" / "manifest.json").read_text(encoding="utf-8"))[
        "memory_guard"
    ]
    assert guard["runtime_cap"]["share_bytes"] == budget // 2
    assert guard["runtime_cap"]["soft_limit_bytes"] == ranks[0]["during"][0]
    # Two replicas of the weights and one row on each rank.
    single = preflight(config, settings, world_size=1, strategy="none", budget=budget)
    assert guard["estimate"]["rows_in_flight"] == 2 * single["rows_in_flight"]
    assert (
        guard["estimate"]["components"]["weights and gradients"]
        == 2 * single["components"]["weights and gradients"]
    )
