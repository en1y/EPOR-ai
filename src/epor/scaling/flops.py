"""Analytic parameter and FLOP counts for EPOR decoders (no PyTorch, no allocation).

Formulation (``disclosed``, Kaplan et al. 2020, arXiv:2001.08361, Table 1):
per token, a decoder's forward pass costs ``2 N + 2 n_layer n_ctx d_attn``
FLOPs, where ``N`` is the number of weights used in matrix multiplies and the
second term is the attention score and value products; the backward pass costs
about twice the forward, so training costs ``3 x forward ~= 6 N + 6 n_layer
n_ctx d_attn`` per token, which for long runs is the familiar ``C ~= 6 N D``.

EPOR adaptations (``EPOR-adaptation``):

* ``N`` here counts every matmul weight *including the output-head logit
  projection* ``d_model x vocab_size``.  Kaplan's ``N`` excludes embeddings;
  EPOR ties the head to the embedding but still performs that matmul, and at a
  65,536 vocabulary it is a large share of small-model compute.  Norm gains
  and the embedding lookup are excluded (they are not matmuls).
* A mixture-of-experts layer counts its router and only the
  ``experts_per_token + n_shared_experts`` experts a token visits.
* Each multi-token-prediction depth adds its ``2 d -> d`` projection, one
  decoder block, its attention term, and one more full-vocabulary logit
  projection (DeepSeek-V3-style MTP, ``deepseek-v3``, ``disclosed``).
* ``n_ctx`` is the training sequence length.  Like Kaplan's formula this does
  not halve the attention term for causal masking, so it is a slight
  over-count for the reference attention kernel.

Parameter conventions: ``total`` is :func:`structural_parameter_count`;
``non_embedding`` removes the token embedding (and an untied head), which is
the ``N`` the scaling ladder and fits use (Kaplan's convention); ``active``
removes experts a token never visits.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from epor.models.config import ModelConfig, mtp_parameter_count, structural_parameter_count

FLOPS_FORMULA = (
    "train_flops_per_token = 3 * forward; forward = 2 * matmul_parameters "
    "+ 2 * attention_layers * sequence_length * d_model"
)


class ParameterCounts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    total: int
    non_embedding: int
    active: int
    embedding: int
    mtp: int


class FlopsEstimate(BaseModel):
    """Analytic per-token FLOPs of one configuration at one sequence length."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence_length: int
    matmul_parameters: int
    attention_layers: int
    attention_flops_per_token: int
    forward_flops_per_token: int
    train_flops_per_token: int
    mtp_forward_flops_per_token: int
    formula: str = FLOPS_FORMULA


def parameter_counts(config: ModelConfig) -> ParameterCounts:
    total = structural_parameter_count(config)
    embedding = config.vocab_size * config.d_model * (1 if config.tie_embeddings else 2)
    inactive = 0
    if config.architecture == "moe-decoder":
        expert = 3 * config.d_model * config.d_ff
        inactive = config.n_layers * expert * (config.n_routed_experts - config.experts_per_token)
    return ParameterCounts(
        total=total,
        non_embedding=total - embedding,
        active=total - inactive,
        embedding=embedding,
        mtp=mtp_parameter_count(config),
    )


def _block_matmul_parameters(config: ModelConfig) -> int:
    hidden = config.d_model
    key_value_width = config.n_kv_heads * (hidden // config.n_heads)
    attention = 2 * hidden * hidden + 2 * hidden * key_value_width
    expert = 3 * hidden * config.d_ff
    if config.architecture == "moe-decoder":
        feed_forward = hidden * config.n_routed_experts + expert * (
            config.experts_per_token + config.n_shared_experts
        )
    else:
        feed_forward = expert
    return attention + feed_forward


def estimate_flops(config: ModelConfig, sequence_length: int) -> FlopsEstimate:
    """Per-token forward and training FLOPs at ``sequence_length`` (see module docstring)."""

    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    block = _block_matmul_parameters(config)
    logits = config.d_model * config.vocab_size
    main_matmul = config.n_layers * block + logits
    mtp_matmul = config.mtp_depth * (2 * config.d_model * config.d_model + block + logits)
    attention_layers = config.n_layers + config.mtp_depth
    attention = 2 * attention_layers * sequence_length * config.d_model
    mtp_attention = 2 * config.mtp_depth * sequence_length * config.d_model
    forward = 2 * (main_matmul + mtp_matmul) + attention
    return FlopsEstimate(
        sequence_length=sequence_length,
        matmul_parameters=main_matmul + mtp_matmul,
        attention_layers=attention_layers,
        attention_flops_per_token=attention,
        forward_flops_per_token=forward,
        train_flops_per_token=3 * forward,
        mtp_forward_flops_per_token=2 * mtp_matmul + mtp_attention,
    )


__all__ = [
    "FLOPS_FORMULA",
    "FlopsEstimate",
    "ParameterCounts",
    "estimate_flops",
    "parameter_counts",
]
