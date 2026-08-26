from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from epor.models.config import (
    ModelConfig,
    load_model_config,
    structural_parameter_count,
    validate_config,
)

CONFIG_ROOT = Path("configs/models")


def test_public_family_names_and_context_contracts() -> None:
    # Capability descends with the alphabet: α is the highest-capability family.
    expected = {
        "epor-alpha.yaml": ("EPOR-α", "epor-alpha", "moe-decoder", 262_144),
        "epor-beta.yaml": ("EPOR-β", "epor-beta", "dense-decoder", 262_144),
        "epor-gamma.yaml": ("EPOR-γ", "epor-gamma", "dense-decoder", 131_072),
    }
    for filename, (display_name, slug, architecture, context) in expected.items():
        config = validate_config(CONFIG_ROOT / filename)
        assert config.display_name == display_name
        assert config.slug == slug
        assert config.architecture == architecture
        assert config.configured_max_context == context
        assert config.validated_max_context == 0
        assert all(profile.status == "planned" for profile in config.certified_profiles)


def test_gamma_encodes_optional_quarter_width_slice() -> None:
    gamma = load_model_config(CONFIG_ROOT / "epor-gamma.yaml")
    assert gamma.elastic_width_multipliers == (0.25, 0.5, 0.75, 1.0)


def test_meta_parameter_counts_match_declared_targets() -> None:
    pytest.importorskip("torch")
    from epor.models.reference import parameter_report

    for filename in (
        "epor-tiny.yaml",
        "epor-reference.yaml",
        "epor-alpha.yaml",
        "epor-beta.yaml",
        "epor-gamma.yaml",
    ):
        config = load_model_config(CONFIG_ROOT / filename)
        report = parameter_report(config)
        assert report.measured_total_parameters == config.total_parameters
        assert structural_parameter_count(config) == report.measured_total_parameters

    reference = parameter_report(load_model_config(CONFIG_ROOT / "epor-reference.yaml"))
    assert 10_000_000 <= reference.measured_total_parameters <= 50_000_000


def test_invalid_context_evidence_is_rejected() -> None:
    with pytest.raises(ValidationError, match="validated_max_context"):
        ModelConfig(
            display_name="bad",
            slug="bad",
            family="reference",
            vocab_size=260,
            d_model=64,
            n_layers=1,
            n_heads=4,
            n_kv_heads=2,
            d_ff=128,
            configured_max_context=16,
            trained_max_context=8,
            validated_max_context=16,
        )
