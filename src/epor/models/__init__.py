"""EPOR model contracts and a lazily imported PyTorch reference model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import (
    CertifiedProfile,
    ModelConfig,
    dump_model_config,
    load_model_config,
    structural_parameter_count,
    validate_config,
)
from .tokenizer import DebugByteTokenizer

if TYPE_CHECKING:
    from .reference import DecoderOutput, EporDecoder, ParameterReport, parameter_report

__all__ = [
    "CertifiedProfile",
    "DebugByteTokenizer",
    "DecoderOutput",
    "EporDecoder",
    "ModelConfig",
    "ParameterReport",
    "dump_model_config",
    "load_model_config",
    "parameter_report",
    "structural_parameter_count",
    "validate_config",
]


def __getattr__(name: str):
    if name in {"DecoderOutput", "EporDecoder", "ParameterReport", "parameter_report"}:
        from . import reference

        return getattr(reference, name)
    raise AttributeError(name)
