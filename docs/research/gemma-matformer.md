# Gemma 3, Gemma 3n, and MatFormer dossier

Source IDs refer to [`research/catalog.yaml`](../../research/catalog.yaml).
This dossier narrows the inference from Google's releases to the experiments
EPOR can actually perform.

## Gemma 3

- **`[disclosed]`** `gemma-3-technical-report` describes a Gemma 3 family using
  grouped-query attention and a higher proportion of bounded local-attention
  layers interleaved with global-attention layers to manage long-context cache
  cost.
- **`[disclosed]`** The report describes knowledge distillation during training
  and configured context up to 128K for larger Gemma 3 variants.
- **`[disclosed]`** The described pattern places five local sliding-window
  attention layers between each global layer, with a bounded window on the local
  layers; the report's 2B configuration pairs that ratio with a 1024-token
  window.
- **`[reported-result]`** The report compares local-to-global ratios from 1:1
  through 7:1 and observes little perplexity movement across them, and separately
  compares sliding-window sizes; these are results for Google's models and setup.
- **`[disclosed]`** The report replaces the previous generation's attention logit
  soft-capping with query/key normalization.
- **`[disclosed]`** For long context the report raises the RoPE base frequency on
  global layers from ten thousand to one million while leaving local layers at
  ten thousand, and extends position handling by an interpolation-style
  procedure with a small integer scaling factor.
- **`[disclosed]`** Quantized releases are produced by a short quantization-aware
  fine-tune, on the order of a few thousand steps, using the unquantized
  checkpoint's own output probabilities as targets.
- **`[reported-result]`** Google's quality and context measurements apply to its
  released models, data, training, and evaluation setup; EPOR has not reproduced
  them.
- **`[EPOR-adaptation]`** EPOR-α tests backend-supported local/global attention
  against a full-attention reference and records position-wise quality, TTFT,
  throughput, KV-cache memory, and short-context regression before adoption. The
  ratio, window size, dual RoPE base, and QK-norm are each separate proxy-scale
  ablations with a dense control, not settings copied because Gemma uses them.
- **`[hypothesis]`** A ratio that costs little perplexity in Google's study may
  behave differently under EPOR's data mixture, tokenizer, and much smaller proxy
  scales; the KV-cache saving is the reason to test it, not evidence of quality
  parity.

## Gemma 3n parameter meanings

- **`[disclosed]`** `gemma-3n-overview` and
  `gemma-3n-developer-guide` describe E2B/E4B as effective parameter labels, not
  raw checkpoint parameter counts.
- **`[disclosed]`** Google describes the E4B package as roughly eight billion raw
  parameters with an approximately four-billion-parameter core/effective path,
  using Per-Layer Embedding caching and conditional parameter loading to keep
  some parameters outside accelerator-resident execution.
- **`[disclosed]`** Google describes E4B as containing the E2B nested model and
  describes intermediate submodel selection through its MatFormer structure.
- **`[unknown]`** Public overview material is not a complete reproducible Gemma
  3n data recipe, optimizer schedule, training curriculum, PLE implementation,
  or runtime for arbitrary hardware.
- **`[EPOR-adaptation]`** EPOR-γ's “≈4B core” denotes an intended accelerator-
  resident path. It is not a statement that an ≈8B dense checkpoint becomes a
  4B-active MoE, nor a claim that EPOR already implements Gemma 3n.

## MatFormer

- **`[disclosed]`** `matformer` nests smaller feed-forward widths inside larger
  Transformer blocks and jointly optimizes a fixed set of granularities so that
  multiple submodels share parameters.
- **`[reported-result]`** The paper reports decoder and encoder experiments up to
  its studied scales, including extracted intermediate models and speculative-
  decoding benefits; those results do not prove behavior at EPOR-γ scale.
- **`[hypothesis]`** Joint slice training may yield a useful quality/latency
  continuum and make a smaller draft/core share behavior with the full model,
  but gradient interference may weaken one or more slices.
- **`[EPOR-adaptation]`** Proxy experiments train every declared slice, compare
  each with an independently trained compute-matched dense model, measure slice
  gradient/update coverage, and test static extraction by comparing logits.

## EPOR-γ gate

- **`[EPOR-adaptation]`** The destination is about 8B total parameters, a nested
  ≈4B core, and an optional ≈2B slice, trained from random initialization with
  licensed multi-teacher distillation signals where available.
- **`[EPOR-adaptation]`** A conventional standalone ≈4B GGUF artifact is required
  even if dynamic PLE execution is delayed; users must not need a bespoke
  streaming runtime to obtain the core local model.
- **`[hypothesis]`** PLE or Engram-style conditional memory is useful only if it
  improves knowledge retained per active FLOP, resident GB, and latency against
  dense/MatFormer controls without unacceptable TTFT or storage traffic.
- **`[EPOR-adaptation]`** Promotion requires slice-training coverage, extraction
  logit parity, quality/compute curves, quantization parity, measured local
  memory and speed, and 128K cloud/32K reference-machine context certification.

## Scope boundary

- **`[disclosed]`** Gemma 3n supports multimodal inputs, but its text path is only
  one part of the released package.
- **`[EPOR-adaptation]`** EPOR remains text-only through v1. Vision/audio
  parameters, benchmarks, or branding are not imported into γ.
- **`[unknown]`** Until proxy experiments pass, whether nested slices plus
  conditional memory outperform a straightforward dense model on EPOR's
  hardware and data remains unknown.
