"""Key/value caches for incremental decoding, with bounded windows and int8 storage.

One :class:`LayerCache` per decoder layer holds what that layer's attention
needs to attend to earlier tokens without recomputing them:

* **full** layers keep every key and value;
* **local** (sliding-window) layers keep only the last ``window`` tokens — a
  bounded ring, so their memory stops growing at the window size;
* **MLA** layers keep the shared ``kv_rank`` latent and the shared positional
  key instead of per-head keys and values (``deepseek-v2``, ``disclosed``);
* layers with a sparse-attention indexer also keep each token's index key.

Keys are cached *after* RoPE, at their absolute positions, so a cached key is
valid for every later query.  With ``quantize="int8"`` every cached tensor is
stored as int8 with one FP32 absmax scale per token and head (per token for
the MLA latent), and dequantized on read.  :func:`kv_cache_bytes` gives the
matching analytic size; :meth:`DecoderCache.nbytes` measures the real one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor

from .config import ModelConfig

Quantization = Literal["none", "int8"]


def quantize_int8(values: Tensor) -> tuple[Tensor, Tensor]:
    """Symmetric absmax int8 over the last dimension: ``(int8 values, FP32 scales)``."""

    scale = values.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 127.0
    quantized = torch.clamp(torch.round(values.float() / scale), -127, 127).to(torch.int8)
    return quantized, scale


def dequantize_int8(values: Tensor, scale: Tensor, dtype: torch.dtype) -> Tensor:
    return (values.float() * scale).to(dtype)


@dataclass
class _Stored:
    """One cached stream: ``[batch, heads, time, dim]`` (heads may be 1)."""

    quantize: Quantization
    values: Tensor | None = None
    scales: Tensor | None = None

    def append(self, item: Tensor, keep: int | None) -> None:
        if self.quantize == "int8":
            quantized, scale = quantize_int8(item)
            self.values = _cat(self.values, quantized, keep)
            self.scales = _cat(self.scales, scale, keep)
        else:
            self.values = _cat(self.values, item, keep)

    def read(self, dtype: torch.dtype) -> Tensor:
        assert self.values is not None
        if self.quantize == "int8":
            assert self.scales is not None
            return dequantize_int8(self.values, self.scales, dtype)
        return self.values

    def nbytes(self) -> int:
        total = 0
        for tensor in (self.values, self.scales):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
        return total


def _cat(existing: Tensor | None, item: Tensor, keep: int | None) -> Tensor:
    joined = item if existing is None else torch.cat((existing, item), dim=-2)
    if keep is not None and joined.shape[-2] > keep:
        joined = joined[..., -keep:, :].contiguous()
    return joined


@dataclass
class LayerCache:
    window: int | None
    quantize: Quantization = "none"
    streams: dict[str, _Stored] = field(default_factory=dict)
    positions: Tensor | None = None

    def update(self, start: int, items: dict[str, Tensor]) -> tuple[dict[str, Tensor], Tensor]:
        """Append ``items`` for positions ``start..``; return visible items and their positions.

        A local layer must still see the tokens being appended together, so it
        keeps ``window - 1`` older tokens plus every new one, then trims to the
        window after the call (the ring never exceeds ``window`` between steps).
        """

        count = next(iter(items.values())).shape[-2]
        new_positions = torch.arange(start, start + count)
        keep = None if self.window is None else self.window - 1 + count
        for name, item in items.items():
            self.streams.setdefault(name, _Stored(self.quantize)).append(item, keep)
        self.positions = _cat(
            None if self.positions is None else self.positions[:, None],
            new_positions[:, None],
            keep,
        )[:, 0]
        visible = {name: self.streams[name].read(item.dtype) for name, item in items.items()}
        positions = self.positions
        if self.window is not None:
            # Trim to the ring size once the new tokens have been attended.
            for stream in self.streams.values():
                stream.values = _trim(stream.values, self.window)
                stream.scales = _trim(stream.scales, self.window)
            self.positions = self.positions[-self.window :]
        return visible, positions

    def nbytes(self) -> int:
        return sum(stream.nbytes() for stream in self.streams.values())


def _trim(tensor: Tensor | None, keep: int) -> Tensor | None:
    if tensor is None or tensor.shape[-2] <= keep:
        return tensor
    return tensor[..., -keep:, :].contiguous()


@dataclass
class DecoderCache:
    """All layers' caches plus the number of tokens already processed."""

    layers: list[LayerCache]
    length: int = 0

    @classmethod
    def for_config(cls, config: ModelConfig, *, quantize: Quantization = "none") -> DecoderCache:
        return cls(
            layers=[
                LayerCache(
                    window=config.sliding_window if config.layer_is_local(layer) else None,
                    quantize=quantize,
                )
                for layer in range(config.n_layers)
            ]
        )

    def nbytes(self) -> int:
        return sum(layer.nbytes() for layer in self.layers)


def kv_cache_bytes(
    config: ModelConfig,
    context_length: int,
    *,
    batch: int = 1,
    dtype_bytes: int = 2,
    quantize: Quantization = "none",
) -> int:
    """Analytic cache size after ``context_length`` tokens (activations excluded).

    ``dtype_bytes`` is the storage width of an unquantized element (2 for
    BF16/FP16, 4 for FP32).  int8 storage costs one byte per element plus one
    FP32 scale per token and head (per token for the MLA latent and positional
    key and the index key).
    """

    head = config.d_model // config.n_heads
    element = 1 if quantize == "int8" else dtype_bytes
    scale = 4 if quantize == "int8" else 0
    total = 0
    for layer in range(config.n_layers):
        tokens = context_length
        if config.layer_is_local(layer):
            tokens = min(context_length, config.sliding_window)
        if config.attention_variant == "mla":
            per_token = (config.mla_kv_rank + config.mla_rope_dim) * element + 2 * scale
        else:
            per_token = 2 * config.n_kv_heads * (head * element + scale)
        if config.sparse_attention is not None:
            per_token += config.sparse_attention.index_dim * element + scale
        total += tokens * per_token
    return total * batch


__all__ = [
    "DecoderCache",
    "LayerCache",
    "Quantization",
    "dequantize_int8",
    "kv_cache_bytes",
    "quantize_int8",
]
