"""Pure-PyTorch reference decoder used for correctness and CPU smoke tests.

Packed-sequence convention
--------------------------
``EporDecoder.forward`` accepts optional ``segment_ids`` of shape
``[batch, sequence]``.  Each document occupies one *contiguous* run of equal
IDs.  A token attends only to earlier tokens with the same ID, and RoPE
positions restart at zero at the start of every run, so a packed document's
logits equal running that document alone (within the attention tolerances
pinned in the tests).  Padding is just another run; give its targets
``IGNORE_INDEX``.

Targets follow PyTorch's cross-entropy convention: ``IGNORE_INDEX`` (-100)
excludes a position from the loss.  The model cannot know which next token a
caller paired with a document's final position, so the data pipeline must set
that target to EOS (when the document really ends there) or ``IGNORE_INDEX``.
Multi-token-prediction targets are masked by the model itself whenever they
would cross a run boundary.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal, cast

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig, mtp_logit_bytes_per_token

IGNORE_INDEX = -100
AttentionBackend = Literal["sdpa", "math"]


@dataclass(slots=True)
class DecoderOutput:
    """Model outputs.

    ``loss`` is the optimized objective: ``main_loss`` plus
    ``mtp_loss_weight`` times the mean of ``mtp_losses``.  Without MTP it is
    the same tensor as ``main_loss``.  Report ``main_loss`` for evaluation and
    perplexity; it is the quantity comparable with single-token training.
    """

    logits: Tensor
    loss: Tensor | None = None
    main_loss: Tensor | None = None
    mtp_losses: tuple[Tensor, ...] = field(default_factory=tuple)


class ParameterReport(BaseModel):
    """Measured implementation counts beside declared architecture targets."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    measured_total_parameters: int
    estimated_active_parameters: int
    declared_total_parameters: int | None
    declared_active_parameters: int | None
    declared_core_parameters: int | None
    # Training-only multi-token-prediction overhead, already included in the
    # measured total.  MTP modules can be dropped for inference.
    mtp_parameters: int = 0
    mtp_logit_bytes_per_token: int = 0


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


def apply_rope(
    query: Tensor,
    key: Tensor,
    *,
    theta: float,
    positions: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Apply rotary position embeddings to ``[batch, heads, time, dim]`` tensors.

    ``positions`` is an optional integer ``[batch, time]`` tensor; without it
    every row uses ``0..time-1``.
    """

    if query.shape[-2:] != key.shape[-2:]:
        raise ValueError("query and key must have matching sequence and head dimensions")
    sequence_length = query.shape[-2]
    head_dimension = query.shape[-1]
    dimensions = torch.arange(0, head_dimension, 2, device=query.device, dtype=torch.float32)
    inverse_frequency = theta ** (-dimensions / head_dimension)
    if positions is None:
        steps = torch.arange(sequence_length, device=query.device, dtype=torch.float32)
        angles = torch.outer(steps, inverse_frequency)
        angles = torch.repeat_interleave(angles, 2, dim=-1)[None, None, :, :]
    else:
        if positions.shape != (query.shape[0], sequence_length):
            raise ValueError("positions must have shape [batch, sequence]")
        angles = positions.to(torch.float32)[..., None] * inverse_frequency
        angles = torch.repeat_interleave(angles, 2, dim=-1)[:, None, :, :]
    cosine = angles.cos().to(query.dtype)
    sine = angles.sin().to(query.dtype)
    return (
        query * cosine + _rotate_half(query) * sine,
        key * cosine + _rotate_half(key) * sine,
    )


def segment_positions(segment_ids: Tensor) -> Tensor:
    """Return RoPE positions that restart at zero at every contiguous segment run."""

    sequence = segment_ids.shape[1]
    steps = torch.arange(sequence, device=segment_ids.device).expand(segment_ids.shape)
    starts = torch.ones(segment_ids.shape, dtype=torch.bool, device=segment_ids.device)
    starts[:, 1:] = segment_ids[:, 1:] != segment_ids[:, :-1]
    run_start = torch.where(starts, steps, torch.zeros_like(steps)).cummax(dim=1).values
    return steps - run_start


def segment_attention_mask(segment_ids: Tensor) -> Tensor:
    """Boolean ``[batch, 1, query, key]`` mask: causal and within one segment."""

    sequence = segment_ids.shape[1]
    causal = torch.ones(sequence, sequence, dtype=torch.bool, device=segment_ids.device).tril()
    same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
    return (same_segment & causal)[:, None, :, :]


def math_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Explicit matmul/mask/softmax attention used only to verify SDPA.

    Inputs are ``[batch, heads, time, dim]`` with key/value heads already
    repeated for GQA.  Scores, softmax, and the value product run in FP32 and
    the result is cast back to the input dtype.  ``attention_mask`` is boolean
    with True meaning "may attend"; without one a plain causal mask is used.
    """

    sequence = query.shape[-2]
    if attention_mask is None:
        attention_mask = torch.ones(sequence, sequence, dtype=torch.bool, device=query.device)
        attention_mask = attention_mask.tril()
    scores = query.float() @ key.float().transpose(-2, -1) / math.sqrt(query.shape[-1])
    probabilities = scores.masked_fill(~attention_mask, float("-inf")).softmax(dim=-1)
    return (probabilities @ value.float()).to(query.dtype)


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
        # QK-norm normalizes each head's query and key vectors after projection
        # and *before* RoPE, the placement Gemma 3 describes
        # (``gemma-3-technical-report``).  Normalizing before the rotation bounds
        # the attention-logit scale while leaving the relative-position rotation
        # itself untouched.  One weight vector per projection is shared across
        # heads.  Without ``qk_norm`` the block is the pre-norm control.
        self.query_norm: RMSNorm | None = None
        self.key_norm: RMSNorm | None = None
        if config.qk_norm:
            self.query_norm = RMSNorm(
                self.head_dimension, config.rms_norm_eps, device=device, dtype=dtype
            )
            self.key_norm = RMSNorm(
                self.head_dimension, config.rms_norm_eps, device=device, dtype=dtype
            )
        # "math" exists only so tests can check SDPA against an explicit
        # reference; training and inference always use "sdpa".
        self.backend: AttentionBackend = "sdpa"

    def forward(
        self,
        inputs: Tensor,
        positions: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        batch, sequence, _ = inputs.shape
        query = self.query(inputs).view(batch, sequence, self.n_heads, self.head_dimension)
        key = self.key(inputs).view(batch, sequence, self.n_kv_heads, self.head_dimension)
        value = self.value(inputs).view(batch, sequence, self.n_kv_heads, self.head_dimension)
        if self.query_norm is not None and self.key_norm is not None:
            query = self.query_norm(query)
            key = self.key_norm(key)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = apply_rope(query, key, theta=self.rope_theta, positions=positions)
        if self.groups != 1:
            key = key.repeat_interleave(self.groups, dim=1)
            value = value.repeat_interleave(self.groups, dim=1)
        if self.backend == "math":
            if self.training and self.dropout:
                raise ValueError("math attention is a verification path without dropout")
            attended = math_attention(query, key, value, attention_mask=attention_mask)
        else:
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=attention_mask is None,
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

    def forward(
        self,
        inputs: Tensor,
        positions: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        hidden = inputs + self.attention(self.attention_norm(inputs), positions, attention_mask)
        return hidden + self.feed_forward(self.feed_forward_norm(hidden))


class MultiTokenPredictionModule(nn.Module):
    """One sequential MTP depth in the style of DeepSeek-V3 (``deepseek-v3``).

    Depth ``k`` combines the previous depth's hidden state at position ``i``
    with the embedding of token ``i + k`` as
    ``projection([RMSNorm(hidden); RMSNorm(embedding)])``, runs one decoder
    block, and predicts token ``i + k + 1`` through the shared output head.
    ``output_norm`` is this depth's own final norm before that shared head
    (``EPOR-adaptation``: it mirrors the main model's ``final_norm``).
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        eps = config.rms_norm_eps
        self.hidden_norm = RMSNorm(config.d_model, eps, device=device, dtype=dtype)
        self.embedding_norm = RMSNorm(config.d_model, eps, device=device, dtype=dtype)
        self.projection = nn.Linear(
            2 * config.d_model, config.d_model, bias=False, device=device, dtype=dtype
        )
        self.block = DecoderBlock(config, device=device, dtype=dtype)
        self.output_norm = RMSNorm(config.d_model, eps, device=device, dtype=dtype)

    def combine(self, previous_hidden: Tensor, embeddings: Tensor) -> Tensor:
        normalized = (self.hidden_norm(previous_hidden), self.embedding_norm(embeddings))
        return self.projection(torch.cat(normalized, dim=-1))


class EporDecoder(nn.Module):
    """Decoder-only EPOR reference implementation.

    ``activation_checkpointing`` is a runtime memory/compute trade, not an
    architecture choice: it recomputes each decoder block's activations during
    backward (``torch.utils.checkpoint`` with ``use_reentrant=False``) and does
    not change the math.  It therefore lives on the module rather than in
    :class:`ModelConfig` and never alters a config hash.  It only takes effect
    in training mode with gradients enabled.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        seed: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        initialize: bool = True,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.activation_checkpointing = activation_checkpointing
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
            # Registered last so the default (mtp_depth=0) parameter order and
            # seeded initialization stay identical to the single-token model.
            self.mtp_modules = nn.ModuleList(
                [
                    MultiTokenPredictionModule(config, device=device, dtype=dtype)
                    for _ in range(config.mtp_depth)
                ]
            )
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

    def _run_block(
        self,
        block: nn.Module,
        hidden: Tensor,
        positions: Tensor | None,
        attention_mask: Tensor | None,
    ) -> Tensor:
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return cast(
                Tensor,
                checkpoint(block, hidden, positions, attention_mask, use_reentrant=False),
            )
        return block(hidden, positions, attention_mask)

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        *,
        segment_ids: Tensor | None = None,
    ) -> DecoderOutput:
        """Run the decoder; see the module docstring for the packing convention.

        ``targets[:, t]`` is the token that follows ``input_ids[:, t]``.
        ``segment_ids`` restricts attention to contiguous same-ID runs and
        restarts RoPE positions per run.  MTP losses are computed only when
        ``targets`` are given; ``logits`` always come from the main head.
        """

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
        positions: Tensor | None = None
        attention_mask: Tensor | None = None
        if segment_ids is not None:
            if segment_ids.shape != input_ids.shape:
                raise ValueError("segment_ids must have the same shape as input_ids")
            if segment_ids.dtype not in {torch.int32, torch.int64}:
                raise TypeError("segment_ids must contain integer IDs")
            positions = segment_positions(segment_ids)
            attention_mask = segment_attention_mask(segment_ids)
        hidden = self.token_embeddings(input_ids)
        for block in self.blocks:
            hidden = self._run_block(block, hidden, positions, attention_mask)
        logits = self.lm_head(self.final_norm(hidden))
        if targets is None:
            return DecoderOutput(logits=logits)
        if targets.shape != input_ids.shape:
            raise ValueError("targets must have the same shape as input_ids")
        if targets.dtype not in {torch.int32, torch.int64}:
            raise TypeError("targets must contain integer token IDs")
        main_loss = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        mtp_losses = self._mtp_losses(hidden, input_ids, targets, segment_ids)
        loss = main_loss
        if mtp_losses:
            loss = main_loss + self.config.mtp_loss_weight * torch.stack(mtp_losses).mean()
        return DecoderOutput(logits=logits, loss=loss, main_loss=main_loss, mtp_losses=mtp_losses)

    def _mtp_losses(
        self,
        hidden: Tensor,
        input_ids: Tensor,
        targets: Tensor,
        segment_ids: Tensor | None,
    ) -> tuple[Tensor, ...]:
        """Sequential DeepSeek-V3-style MTP losses, one per depth.

        At depth ``k`` position ``i`` sees the depth ``k - 1`` hidden state at
        ``i`` and the embedding of ``input_ids[i + k]`` and is trained on
        ``targets[i + k]`` (token ``i + k + 1``).  The chain keeps full causal
        attention over the ``sequence - k`` usable positions.  Targets whose
        position ``i + k`` lies in a different segment than ``i`` are ignored.
        """

        if not self.mtp_modules:
            return ()
        sequence = input_ids.shape[1]
        if sequence <= self.config.mtp_depth:
            raise ValueError("sequence length must exceed mtp_depth to compute MTP losses")
        losses: list[Tensor] = []
        previous = hidden
        for depth, module in enumerate(self.mtp_modules, start=1):
            module = cast(MultiTokenPredictionModule, module)
            length = sequence - depth
            combined = module.combine(
                previous[:, :length], self.token_embeddings(input_ids[:, depth:])
            )
            depth_targets = targets[:, depth:]
            positions: Tensor | None = None
            attention_mask: Tensor | None = None
            if segment_ids is not None:
                anchor_segments = segment_ids[:, :length]
                positions = segment_positions(anchor_segments)
                attention_mask = segment_attention_mask(anchor_segments)
                crosses = segment_ids[:, depth:] != anchor_segments
                depth_targets = depth_targets.masked_fill(crosses, IGNORE_INDEX)
            previous = self._run_block(module.block, combined, positions, attention_mask)
            logits = self.lm_head(module.output_norm(previous))
            flat_targets = depth_targets.reshape(-1)
            # Sum then divide so a depth with no valid target yields zero, not NaN.
            total = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                flat_targets,
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            valid = (flat_targets != IGNORE_INDEX).sum().clamp(min=1)
            losses.append(total / valid)
        return tuple(losses)

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


@contextmanager
def attention_backend(model: nn.Module, backend: AttentionBackend) -> Iterator[None]:
    """Temporarily switch every attention module in ``model`` to ``backend``.

    Verification-only: the explicit "math" path exists to check SDPA.
    """

    modules = [module for module in model.modules() if isinstance(module, GroupedQueryAttention)]
    previous: list[AttentionBackend] = [module.backend for module in modules]
    for module in modules:
        module.backend = backend
    try:
        yield
    finally:
        for module, value in zip(modules, previous, strict=True):
            module.backend = value


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
        mtp_parameters=sum(parameter.numel() for parameter in model.mtp_modules.parameters()),
        mtp_logit_bytes_per_token=mtp_logit_bytes_per_token(config),
    )
