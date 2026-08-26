from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from epor.reference_policy import (
    REFERENCE_CHECKPOINT_MAX_BYTES,
    REFERENCE_CORPUS_MAX_BYTES,
    REFERENCE_MAX_BATCH_TOKENS,
    REFERENCE_MAX_EVAL_BATCHES,
    REFERENCE_MAX_NEW_TOKENS,
    REFERENCE_MAX_STEPS,
)

torch = pytest.importorskip("torch")

from epor.training import evaluate, generate, load_checkpoint, pretrain  # noqa: E402
from epor.training.data import load_token_corpus  # noqa: E402

TINY_CONFIG = Path("configs/models/epor-tiny.yaml")


def test_oversized_corpus_is_rejected_before_it_is_loaded(tmp_path: Path) -> None:
    corpus = tmp_path / "oversized.txt"
    with corpus.open("wb") as handle:
        handle.truncate(REFERENCE_CORPUS_MAX_BYTES + 1)

    with pytest.raises(ValueError, match="corpus exceeds"):
        load_token_corpus(corpus)


def test_oversized_checkpoint_is_rejected_before_torch_load(tmp_path: Path) -> None:
    checkpoint = tmp_path / "oversized.pt"
    with checkpoint.open("wb") as handle:
        handle.truncate(REFERENCE_CHECKPOINT_MAX_BYTES + 1)

    with pytest.raises(ValueError, match="checkpoint exceeds"):
        load_checkpoint(checkpoint)


def test_training_limits_fail_before_creating_output(tmp_path: Path) -> None:
    steps_output = tmp_path / "too-many-steps"
    with pytest.raises(ValueError, match="max_steps must be between"):
        pretrain(TINY_CONFIG, steps_output, max_steps=REFERENCE_MAX_STEPS + 1)
    assert not steps_output.exists()

    batch_output = tmp_path / "too-many-batch-tokens"
    with pytest.raises(ValueError, match=r"batch_size \* sequence_length"):
        pretrain(
            TINY_CONFIG,
            batch_output,
            max_steps=1,
            batch_size=REFERENCE_MAX_BATCH_TOKENS + 1,
            sequence_length=1,
        )
    assert not batch_output.exists()


def test_oversized_model_recipe_fails_before_creating_output(tmp_path: Path) -> None:
    payload = yaml.safe_load(TINY_CONFIG.read_text(encoding="utf-8"))
    payload["n_layers"] = 2_000
    config = tmp_path / "oversized.yaml"
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    output = tmp_path / "oversized-model"

    with pytest.raises(ValueError, match="50,000,000-parameter"):
        pretrain(config, output, max_steps=1)
    assert not output.exists()


def test_evaluation_and_generation_limits_precede_checkpoint_loading(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pt"
    with pytest.raises(ValueError, match="max_batches must be between"):
        evaluate(missing, max_batches=REFERENCE_MAX_EVAL_BATCHES + 1)
    with pytest.raises(ValueError, match="max_new_tokens must be between"):
        generate(missing, "prompt", max_new_tokens=REFERENCE_MAX_NEW_TOKENS + 1)
