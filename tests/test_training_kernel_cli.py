"""The training-kernel CLI: covenant-gated, exit codes by failure kind, exact resume."""

from __future__ import annotations

import json
import os
import shutil
import signal
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

from click import unstyle
from conftest import TokenShardFixture, make_token_shards
from typer.testing import CliRunner

from epor.cli import app
from epor.control.authority import AuthorityStore

runner = CliRunner()
TINY = Path(__file__).resolve().parents[1] / "configs" / "models" / "epor-tiny.yaml"


@pytest.fixture
def cli_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("EPOR_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("EPOR_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("EPOR_CONTROL_SAFETY_ROOT", str(tmp_path / "safety"))
    monkeypatch.setenv("EPOR_CONTROL_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("EPOR_CONTROL_DATABASE", str(tmp_path / "control.sqlite3"))
    (tmp_path / "configs").mkdir()
    shutil.copyfile(TINY, tmp_path / "configs" / "epor-tiny.yaml")
    return tmp_path


@pytest.fixture
def shards(cli_root: Path) -> TokenShardFixture:
    return make_token_shards(cli_root, max_shard_tokens=4096)


def _train_args(tokenized_id: str, output: str, *extra: str) -> list[str]:
    return [
        "train",
        "kernel",
        "configs/epor-tiny.yaml",
        "--tokenized-id",
        tokenized_id,
        "--output",
        output,
        "--vocab-from-build",
        "--precision",
        "fp32",
        "--seq-len",
        "16",
        "--tokens-per-batch",
        "64",
        "--micro-batch-size",
        "2",
        "--learning-rate",
        "3e-3",
        "--checkpoint-every",
        "2",
        "--eval-max-steps",
        "2",
        *extra,
    ]


def _json(arguments: list[str]) -> dict[str, Any]:
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_kernel_commands_are_executable_typed_surfaces() -> None:
    train = unstyle(runner.invoke(app, ["train", "--help"]).stdout)
    assert "kernel" in train and "kernel-resume" in train and "pretrain" in train
    assert "kernel" in unstyle(runner.invoke(app, ["eval", "--help"]).stdout)


def test_recipe_vocabulary_is_never_resized_silently(
    cli_root: Path, shards: TokenShardFixture
) -> None:
    recipe = cli_root / "configs" / "epor-tiny.yaml"
    before = recipe.read_bytes()
    arguments = _train_args(shards.tokenized_id, "runs/refused", "--max-steps", "2")
    arguments.remove("--vocab-from-build")
    result = runner.invoke(app, arguments)
    assert result.exit_code == 2
    assert "--vocab-from-build" in unstyle(result.stdout)
    assert recipe.read_bytes() == before
    assert not (cli_root / "runs" / "refused").exists()


def test_train_resume_and_evaluate_continue_one_lineage(
    cli_root: Path, shards: TokenShardFixture
) -> None:
    # The schedule spans the whole planned run; it is trajectory, so it is fixed
    # at the start and resuming cannot stretch it.
    schedule = ("--total-steps", "4")
    first = _json(_train_args(shards.tokenized_id, "runs/kernel", "--max-steps", "2", *schedule))
    assert first["steps_completed"] == 2 and first["start_step"] == 0
    assert first["model"]["vocab_size"] == shards.vocab_size
    assert first["model"]["config_sha256"] != ""
    assert first["distributed"] == {"strategy": "none", "world_size": 1}
    assert first["recipe"] == "configs/epor-tiny.yaml"

    # Changing only a runtime setting is allowed and continues the same lineage.
    resumed = _json(
        ["train", "kernel-resume", "runs/kernel", "--max-steps", "4", "--checkpoint-every", "1"]
    )
    assert resumed["start_step"] == 2 and resumed["steps_completed"] == 4
    assert resumed["lineage_id"] == first["lineage_id"]
    assert resumed["run_id"] != first["run_id"]

    reference = _json(
        _train_args(shards.tokenized_id, "runs/reference", "--max-steps", "4", *schedule)
    )
    assert list(resumed["losses"]) == list(reference["losses"][2:])

    stale = runner.invoke(app, ["train", "kernel-resume", "runs/kernel", "--max-steps", "4"])
    assert stale.exit_code == 2

    evaluated = _json(["eval", "kernel", "runs/kernel"])
    assert evaluated["evaluation"]["split"] == "validation"
    assert evaluated["evaluation"]["main_loss"] > 0
    assert "proxy-scale" in evaluated["scope"]
    assert Path(evaluated["report_path"]).is_file()

    actions = [
        record["action_id"]
        for record in AuthorityStore(cli_root / "safety").records()
        if record.get("kind") == "safety.decision"
    ]
    assert {"kernel_train", "kernel_evaluate"} <= set(actions)


def test_local_ddp_run_matches_a_single_process_run(
    cli_root: Path, shards: TokenShardFixture
) -> None:
    single = _json(_train_args(shards.tokenized_id, "runs/single", "--max-steps", "2"))
    ddp = _json(
        _train_args(
            shards.tokenized_id,
            "runs/ddp",
            "--max-steps",
            "2",
            "--micro-batch-size",
            "1",
            "--strategy",
            "ddp",
            "--world-size",
            "2",
        )
    )
    assert ddp["distributed"]["strategy"] == "ddp"
    assert ddp["distributed"]["world_size"] == 2
    assert ddp["losses"] == pytest.approx(single["losses"], abs=1e-6)


@pytest.mark.parametrize(
    "extra",
    [
        ["--strategy", "ddp"],
        ["--world-size", "2"],
        ["--output", "../outside"],
    ],
)
def test_invalid_topology_or_paths_exit_two(
    cli_root: Path, shards: TokenShardFixture, extra: list[str]
) -> None:
    result = runner.invoke(
        app, [*_train_args(shards.tokenized_id, "runs/bad", "--max-steps", "2"), *extra]
    )
    assert result.exit_code == 2, result.output
    assert not (cli_root / "runs" / "bad").exists()


def test_missing_inputs_exit_cleanly(cli_root: Path) -> None:
    missing = runner.invoke(app, _train_args("0" * 64, "runs/missing", "--max-steps", "2"))
    assert missing.exit_code == 2
    assert missing.exception is None or isinstance(missing.exception, SystemExit)
    assert runner.invoke(app, ["eval", "kernel", "runs/none"]).exit_code == 2
    assert runner.invoke(app, ["train", "kernel-resume", "/", "--max-steps", "3"]).exit_code == 2


def test_a_stopped_run_exits_130(
    cli_root: Path, shards: TokenShardFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    from epor.training import TrainingCancelled
    from epor.training import runner as kernel_runner

    def cancelled(*_args: Any, **_kwargs: Any) -> Any:
        raise TrainingCancelled("training cancellation requested")

    monkeypatch.setattr(kernel_runner, "pretrain_packed", cancelled)
    result = runner.invoke(app, _train_args(shards.tokenized_id, "runs/stop", "--max-steps", "2"))
    assert result.exit_code == 130
    assert "training stopped" in result.stderr


def test_first_signal_requests_a_stop_and_the_second_aborts() -> None:
    from epor.cli import _StopRequests  # pyright: ignore[reportPrivateUsage]

    requested: list[bool] = []
    with _StopRequests(lambda: requested.append(True)):
        os.kill(os.getpid(), signal.SIGTERM)
        assert requested == [True]
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGINT)
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


@pytest.mark.parametrize(
    ("action_id", "arguments"),
    [
        ("kernel_train", ["train", "kernel", "configs/epor-tiny.yaml", "--tokenized-id", "x"]),
        ("kernel_train", ["train", "kernel-resume", "runs/x", "--max-steps", "2"]),
        ("kernel_evaluate", ["eval", "kernel", "runs/x"]),
    ],
)
def test_kernel_commands_are_gated_by_the_safety_covenant(
    cli_root: Path, action_id: str, arguments: list[str]
) -> None:
    from epor import actions

    if arguments[:2] == ["train", "kernel"]:
        arguments = [*arguments, "--output", "runs/gated", "--max-steps", "2"]
    withdrawn = actions.DECLARED_ACTIONS.pop(action_id)
    try:
        result = runner.invoke(app, arguments)
    finally:
        actions.DECLARED_ACTIONS[action_id] = withdrawn

    assert result.exit_code == 3
    assert "safety covenant" in result.stdout
    assert not (cli_root / "runs").exists()


def test_derived_models_declare_their_own_structure(shards: TokenShardFixture) -> None:
    from epor.models.config import load_model_config, structural_parameter_count
    from epor.training.runner import derive_kernel_model

    recipe = load_model_config(TINY)
    derived = derive_kernel_model(
        TINY,
        data_root=shards.data_root,
        tokenized_id=shards.tokenized_id,
        vocab_from_build=True,
        qk_norm=True,
        mtp_depth=1,
    )
    assert (derived.vocab_size, derived.qk_norm, derived.mtp_depth) == (shards.vocab_size, True, 1)
    assert derived.total_parameters == structural_parameter_count(derived)
    assert derived.sha256 != recipe.sha256
    assert load_model_config(TINY).sha256 == recipe.sha256


def test_resume_changes_only_runtime_settings(cli_root: Path, shards: TokenShardFixture) -> None:
    from epor.training import CheckpointError
    from epor.training.runner import resume_inputs

    _json(_train_args(shards.tokenized_id, "runs/kernel", "--max-steps", "2"))
    _, settings, run_dir, step = resume_inputs(
        cli_root / "runs" / "kernel", {"keep_checkpoints": 5}
    )
    assert (settings.keep_checkpoints, run_dir, step) == (5, cli_root / "runs" / "kernel", 2)
    with pytest.raises(CheckpointError, match="seq_len"):
        resume_inputs(cli_root / "runs" / "kernel", {"seq_len": 32})
