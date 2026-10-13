from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from epor.models.config import (
    DEFAULT_MTP_LOSS_WEIGHT,
    HASH_NEUTRAL_DEFAULTS,
    ModelConfig,
    load_model_config,
    mtp_logit_bytes_per_token,
    mtp_parameter_count,
    structural_parameter_count,
)

CONFIG_ROOT = Path("configs/models")

# Hashes recorded before QK-norm and MTP fields existed.  Run manifests and
# resume checks compare these values, so adding a default-valued field must
# never move them.
PINNED_CONFIG_SHA256 = {
    "epor-tiny.yaml": "fd9a3df88eb3700dbf449afebcb77f2f9dba6278fec1f0631ea0750ce51a63c9",
    "epor-reference.yaml": "8758885bd7233b89acc3ae010ff594470b4aa6c998e0ed33ae8e27b7c9b7b105",
    "epor-alpha.yaml": "00f628d225780051f145c7970c38684abb2dffdf68e3c0085be434bc7089d615",
    "epor-beta.yaml": "33490f32c3ff2724c7cfc17acbc8b0aa1e5048d833ef9c1da5f1977c4ece143b",
    "epor-gamma.yaml": "47aeb31c72719ed20c2fae0e21620c7aa2a4690c1c277b86768d0a994a31693b",
}


def _tiny(**overrides: Any) -> ModelConfig:
    base = load_model_config(CONFIG_ROOT / "epor-tiny.yaml")
    return ModelConfig.model_validate({**base.model_dump(mode="json"), **overrides})


def test_existing_config_hashes_survive_new_default_fields() -> None:
    for filename, expected in PINNED_CONFIG_SHA256.items():
        config = load_model_config(CONFIG_ROOT / filename)
        assert config.sha256 == expected
        canonical = json.loads(config.canonical_json())
        assert not set(HASH_NEUTRAL_DEFAULTS) & set(canonical)


def test_new_architecture_fields_default_to_the_control_model() -> None:
    config = _tiny()
    assert config.qk_norm is False
    assert config.mtp_depth == 0
    assert config.mtp_loss_weight == DEFAULT_MTP_LOSS_WEIGHT
    assert mtp_parameter_count(config) == 0
    assert mtp_logit_bytes_per_token(config) == 0


def test_non_default_architecture_fields_enter_the_hash() -> None:
    control = _tiny()
    variants = (
        _tiny(qk_norm=True),
        _tiny(mtp_depth=1),
        _tiny(mtp_depth=1, mtp_loss_weight=0.1),
    )
    hashes = {control.sha256, *(variant.sha256 for variant in variants)}
    assert len(hashes) == len(variants) + 1
    assert json.loads(variants[2].canonical_json())["mtp_loss_weight"] == 0.1


def test_mtp_fields_are_validated() -> None:
    with pytest.raises(ValidationError, match="mtp_loss_weight has no effect"):
        _tiny(mtp_loss_weight=0.1)
    with pytest.raises(ValidationError, match="mtp_depth must be smaller"):
        _tiny(mtp_depth=128)
    with pytest.raises(ValidationError):
        _tiny(mtp_depth=-1)
    with pytest.raises(ValidationError):
        _tiny(mtp_depth=1, mtp_loss_weight=1.5)


def test_structural_count_includes_qk_norm_and_mtp_overhead() -> None:
    control = _tiny()
    head_dimension = control.d_model // control.n_heads
    qk = _tiny(qk_norm=True)
    assert structural_parameter_count(qk) - structural_parameter_count(control) == (
        control.n_layers * 2 * head_dimension
    )

    mtp = _tiny(mtp_depth=2)
    embeddings_and_final_norm = control.vocab_size * control.d_model + control.d_model
    block = (structural_parameter_count(control) - embeddings_and_final_norm) // control.n_layers
    # Two input norms, a 2d->d projection, one block, and an output norm per depth.
    per_depth = 2 * control.d_model + 2 * control.d_model**2 + block + control.d_model
    assert mtp_parameter_count(mtp) == 2 * per_depth
    assert structural_parameter_count(mtp) == structural_parameter_count(control) + 2 * per_depth
    assert mtp_logit_bytes_per_token(mtp) == 2 * control.vocab_size * 4
