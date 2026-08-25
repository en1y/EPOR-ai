"""Pure-PyTorch reference decoder used for correctness and CPU smoke tests."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig


@dataclass(slots=True)
class DecoderOutput:
    logits: Tensor
    loss: Tensor | None = None


class ParameterReport(BaseModel):
    """Measured implementation counts beside declared architecture targets."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    measured_total_parameters: int
    estimated_active_parameters: int
    declared_total_parameters: int | None
    declared_active_parameters: int | None
    declared_core_parameters: int | None


class RMSNorm(nn.Module):
    def __init__(
        self,
        dimension: int,
        eps: float = 1e-5,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dimension, device=device, dtype=dtype))

    def forward(self, inputs: Tensor) -> Tensor:
        # Accumulate the variance in FP32 even if an export adapter later uses a
        # lower precision.  The v0.0.1 reference path itself remains FP32.
        variance = inputs.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = inputs * torch.rsqrt(variance.to(inputs.dtype) + self.eps)
        return normalized * self.weight


def _rotate_half(inputs: Tensor) -> Tensor:
    pairs = inputs.float().reshape(*inputs.shape[:-1], -1, 2)
    first, second = pairs.unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).flatten(start_dim=-2).to(inputs.dtype)


def apply_rope(query: Tensor, key: Tensor, *, theta: float) -> tuple[Tensor, Tensor]:
    """Apply rotary position embeddings to ``[batch, heads, time, dim]`` tensors."""

    if query.shape[-2:] != key.shape[-2:]:
        raise ValueError("query and key must have matching sequence and head dimensions")
    sequence_length = query.shape[-2]
    head_dimension = query.shape[-1]
    positions = torch.arange(sequence_length, device=query.device, dtype=torch.float32)
    dimensions = torch.arange(0, head_dimension, 2, device=query.device, dtype=torch.float32)
    inverse_frequency = theta ** (-dimensions / head_dimension)
    angles = torch.outer(positions, inverse_frequency)
    angles = torch.repeat_interleave(angles, 2, dim=-1)[None, None, :, :]
    cosine = angles.cos().to(query.dtype)
    sine = angles.sin().to(query.dtype)
    return (
        query * cosine + _rotate_half(query) * sine,
        key * cosine + _rotate_half(key) * sine,
    )


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dimension = config.d_model // config.n_heads
        self.groups = config.n_heads // config.n_kv_heads
        self.rope_theta = config.rope_theta
        self.dropout = config.dropout
        linear_kwargs = {"bias": False, "device": device, "dtype": dtype}
        self.query = nn.Linear(config.d_model, config.d_model, **linear_kwargs)
        kv_width = self.n_kv_heads * self.head_dimension
        self.key = nn.Linear(config.d_model, kv_width, **linear_kwargs)
        self.value = nn.Linear(config.d_model, kv_width, **linear_kwargs)
        self.output = nn.Linear(config.d_model, config.d_model, **linear_kwargs)

    def forward(self, inputs: Tensor) -> Tensor:
        batch, sequence, _ = inputs.shape
        query = self.query(inputs).view(batch, sequence, self.n_heads, self.head_dimension)
        key = self.key(inputs).view(batch, sequence, self.n_kv_heads, self.head_dimension)
        value = self.value(inputs).view(batch, sequence, self.n_kv_heads, self.head_dimension)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = apply_rope(query, key, theta=self.rope_theta)
        if self.groups != 1:
            key = key.repeat_interleave(self.groups, dim=1)
            value = value.repeat_interleave(self.groups, dim=1)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, -1)
        return self.output(attended)


class SwiGLU(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        kwargs = {"bias": False, "device": device, "dtype": dtype}
        self.gate = nn.Linear(config.d_model, config.d_ff, **kwargs)
        self.up = nn.Linear(config.d_model, config.d_ff, **kwargs)
        self.down = nn.Linear(config.d_ff, config.d_model, **kwargs)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(inputs)) * self.up(inputs))


class SparseMoE(nn.Module):
    """Small, legible top-k MoE reference; not a scale-optimized kernel."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.experts_per_token = config.experts_per_token
        self.router = nn.Linear(
            config.d_model,
            config.n_routed_experts,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.routed_experts = nn.ModuleList(
            [SwiGLU(config, device=device, dtype=dtype) for _ in range(config.n_routed_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [SwiGLU(config, device=device, dtype=dtype) for _ in range(config.n_shared_experts)]
        )

    def forward(self, inputs: Tensor) -> Tensor:
        original_shape = inputs.shape
        flattened = inputs.reshape(-1, original_shape[-1])
        route_probabilities = self.router(flattened).softmax(dim=-1)
        weights, indices = route_probabilities.topk(self.experts_per_token, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        routed = torch.zeros_like(flattened)
        for expert_index, expert in enumerate(self.routed_experts):
            token_positions, slots = torch.where(indices == expert_index)
            if token_positions.numel() == 0:
                continue
            expert_output = expert(flattened.index_select(0, token_positions))
            routed.index_add_(
                0,
                token_positions,
                expert_output * weights[token_positions, slots, None],
            )
        if self.shared_experts:
            shared = torch.stack([expert(flattened) for expert in self.shared_experts]).sum(0)
            routed = routed + shared
        return routed.view(original_shape)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(
            config.d_model, config.rms_norm_eps, device=device, dtype=dtype
        )
        self.attention = GroupedQueryAttention(config, device=device, dtype=dtype)
        self.feed_forward_norm = RMSNorm(
            config.d_model, config.rms_norm_eps, device=device, dtype=dtype
        )
        if config.architecture == "moe-decoder":
            self.feed_forward: nn.Module = SparseMoE(config, device=device, dtype=dtype)
        else:
            self.feed_forward = SwiGLU(config, device=device, dtype=dtype)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = inputs + self.attention(self.attention_norm(inputs))
        return hidden + self.feed_forward(self.feed_forward_norm(hidden))


class EporDecoder(nn.Module):
    """Decoder-only EPOR reference implementation."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        seed: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        initialize: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        # nn.Linear/Embedding constructors normally advance the process-global
        # RNG before ``reset_parameters`` can replace their values.  fork_rng
        # preserves that state, making model construction reproducible without
        # perturbing callers or the exact-resume training stream.
        with torch.random.fork_rng(devices=[]):
            self.token_embeddings = nn.Embedding(
                config.vocab_size,
                config.d_model,
                device=device,
                dtype=dtype,
            )
            self.blocks = nn.ModuleList(
                [DecoderBlock(config, device=device, dtype=dtype) for _ in range(config.n_layers)]
            )
            self.final_norm = RMSNorm(
                config.d_model,
                config.rms_norm_eps,
                device=device,
                dtype=dtype,
            )
            self.lm_head = nn.Linear(
                config.d_model,
                config.vocab_size,
                bias=False,
                device=device,
                dtype=dtype,
            )
            if config.tie_embeddings:
                self.lm_head.weight = self.token_embeddings.weight
            if initialize and not self.token_embeddings.weight.is_meta:
                self.reset_parameters(seed)

    def reset_parameters(self, seed: int) -> None:
        """Initialize all parameters deterministically without consuming global RNG."""

        if seed < 0:
            raise ValueError("seed must be non-negative")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        seen: set[int] = set()
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                if parameter.device.type != "cpu":
                    raise ValueError("v0.0.1 deterministic initialization is CPU-only")
                if parameter.ndim >= 2:
                    parameter.normal_(
                        mean=0.0,
                        std=self.config.initializer_range,
                        generator=generator,
                    )
                elif name.endswith("weight"):
                    parameter.fill_(1.0)
                else:
                    parameter.zero_()
        embedding = self.token_embeddings.weight
        if self.config.pad_token_id is not None:
            with torch.no_grad():
                embedding[self.config.pad_token_id].zero_()

    def forward(self, input_ids: Tensor, targets: Tensor | None = None) -> DecoderOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype not in {torch.int32, torch.int64}:
            raise TypeError("input_ids must contain integer token IDs")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if input_ids.shape[1] > self.config.configured_max_context:
            raise ValueError(
                f"sequence length {input_ids.shape[1]} exceeds configured context "
                f"{self.config.configured_max_context}"
            )
        hidden = self.token_embeddings(input_ids)
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            if targets.dtype not in {torch.int32, torch.int64}:
                raise TypeError("targets must contain integer token IDs")
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )
        return DecoderOutput(logits=logits, loss=loss)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
        eos_token_id: int | None = None,
    ) -> Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if input_ids.ndim != 2 or input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must have non-empty [batch, sequence] shape")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive")
        generated = input_ids
        self.eval()
        for _ in range(max_new_tokens):
            if generated.shape[1] >= self.config.configured_max_context:
                break
            logits = self(generated).logits[:, -1, :]
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                    threshold = values[:, -1, None]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                probabilities = logits.softmax(dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1, generator=generator)
            generated = torch.cat((generated, next_token), dim=1)
            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break
        return generated


def _build_meta_model(config: ModelConfig) -> EporDecoder:
    # The context-manager idiom is the documented PyTorch route for constructing
    # an allocation-free module graph on the meta device.
    with torch.device("meta"):
        return EporDecoder(config, initialize=False)


def parameter_report(config: ModelConfig) -> ParameterReport:
    """Count a target configuration without allocating its parameter storage."""

    model = _build_meta_model(config)
    total = sum(parameter.numel() for parameter in model.parameters())
    if config.architecture == "dense-decoder":
        active = total
    else:
        active = 0
        expert_pattern = re.compile(r"\.routed_experts\.(\d+)\.")
        for name, parameter in model.named_parameters():
            match = expert_pattern.search(name)
            if match and int(match.group(1)) >= config.experts_per_token:
                continue
            active += parameter.numel()
    return ParameterReport(
        model_id=config.slug,
        measured_total_parameters=total,
        estimated_active_parameters=active,
        declared_total_parameters=config.total_parameters,
        declared_active_parameters=config.active_parameters,
        declared_core_parameters=config.core_parameters,
    )
