# Context laboratory (v0.0.8)

This document describes `epor.context` and the attention, positional, and
cache mechanisms it exercises: RoPE scaling and the split RoPE base,
interleaved local/global attention, the KV cache and its int8 quantization,
multi-head latent attention, the sparse-attention retrofit, progressive
context extension in the kernel, the G6 long-context suite, host profiles,
sweeps, and the 32K/128K/256K projection.

**Claim boundary.** Everything here runs on the CPU reference machine at
proxy scale. A tiny model extended from 128 to a few thousand positions and
scored on generated items shows that the mechanisms run, agree with their
references, and fail safely. It is **not** evidence that any EPOR model
handles long context, and 32K, 128K, and 256K appear only as recipes and
projections.

## Three context numbers

Every config, manifest, report, and console view keeps three lengths apart:

| Number | Meaning | Raised by |
| --- | --- | --- |
| Configured | The longest input the code accepts (`configured_max_context`). | Editing a config or `epor context extend --target-context`. |
| Trained | The longest `seq_len` this checkpoint's lineage trained at, recorded in `trainer.json` and the run manifest's `context` section. | Training at that length. |
| Validated | The longest length with a certificate under `<data-root>/context-certificates/`. | Only `epor context certify` after a passing G6 report at that length. |

A model that accepts 4,096 tokens, trained at 1,024, and holds no certificate
has a validated context of 0. Shipped config files keep `trained_max_context`
and `validated_max_context` at 0; measured values live in run artifacts and
certificates, never written back into a recipe.

## Commands and actions

| Command | Covenant action | Job type | Notes |
| --- | --- | --- | --- |
| `epor context extend <ckpt> --target-context N` | `context_extend` | `context_extend` | Continues training at a longer length with RoPE scaling. |
| `epor context retrofit <ckpt>` | `context_extend` | `context_extend` | Adds the sparse-attention indexer. |
| `epor context g6 <ckpt> --length N` | `context_evaluate` | `context_evaluate` | Writes an evaluation report. |
| `epor context profile <ckpt> --length N` | `context_profile` | `context_profile` | TTFT, decode rate, peak RSS, KV bytes, OOM. |
| `epor context feasibility <recipe> --variant V` | `context_profile` | `context_profile` | Largest length that fits, per variant. |
| `epor context sweep <recipe> --axis A` | `context_sweep` | `context_sweep` | Window, ratio, or split RoPE base. |
| `epor context project <config>` | `context_project` | — | Arithmetic only. |
| `epor context certify <ckpt> ...` | `context_certify` | — | Owner CLI only; exit 3 on refusal. |
| `epor context status <ckpt>`, `epor context reports` | `report_review` | — | Read-only. |

Each action is its own delegated scope: training authority does not cover
certification, and evaluation authority does not cover training. Certification
has no job type because a validated context is what later releases rely on.
Stopping is never gated: Ctrl-C checkpoints an extension and exits `130`, and
cancelling a job stops its subprocesses.

## Positional extension

**`[disclosed]`** Position interpolation
([arXiv:2306.15595](https://arxiv.org/abs/2306.15595)), NTK-aware base
scaling, and YaRN ([arXiv:2309.00071](https://arxiv.org/abs/2309.00071))
extend RoPE beyond its training length. **`[EPOR-adaptation]`** All three are
one optional field, `rope_scaling`, applied when the RoPE tables are built:

| `type` | Effect |
| --- | --- |
| `linear` | inverse frequencies divided by `factor` |
| `ntk` | base becomes `rope_theta · factor^(d/(d−2))` |
| `yarn` | each dimension's rotations over `original_max_context` (`r = L·θ_i/2π`) sets a ramp: dimensions with `r ≥ beta_fast` (32) keep their frequency, those with `r ≤ beta_slow` (1) are divided by `factor`, and those between are blended linearly; cos and sin are multiplied by `attention_factor`, default `0.1·ln(factor) + 1` |

`factor` is `target / original_max_context` when `epor context extend` builds
the child config. `--method none` extends with no scaling as a control.

**`[disclosed]`** Gemma 3 ([arXiv:2503.19786](https://arxiv.org/abs/2503.19786))
uses a high RoPE base on global layers and 10,000 on local layers.
**`[EPOR-adaptation]`** `local_rope_theta` gives local layers their own base.
`rope_scaling` applies only to layers using `rope_theta`; a local layer never
sees a distance beyond its window, so it is never scaled.

Every new field (`rope_scaling`, `local_rope_theta`, `sliding_window`,
`local_layers_per_global`, `attention_variant`, `mla_kv_rank`,
`mla_rope_dim`, `sparse_attention`) is in `HASH_NEUTRAL_DEFAULTS`: at its
default it is left out of the config hash, so every shipped config keeps its
v0.0.7 hash and every earlier checkpoint still loads.

## Interleaved local/global attention

With `local_layers_per_global = k` and `sliding_window = w`, in every group of
`k + 1` layers the first `k` attend only to the previous `w` positions
(`i − w < j ≤ i`) and the last attends globally. The window mask composes with
causal and segment masks, so packing still keeps documents apart. MTP blocks
are global. QK-norm and MTP work unchanged.
`test_model_context.py` checks every layer against a math reference built from
the explicit mask at the v0.0.5 tolerance.

## KV cache and int8 quantization

`EporDecoder.forward_cached(input_ids, cache)` runs prefill and incremental
decode on a `DecoderCache`:

- global layers keep every key and value;
- local layers keep a ring of the last `w` positions, so their cache is bounded;
- GQA caches `n_kv_heads` heads;
- MLA caches only its latent and shared positional key.

Prefill-then-decode logits match the no-cache forward to `1e-5` for full GQA,
QK-norm with MTP, local/global, split base with YaRN, MLA, and sparse
selection (`test_model_kv_cache.py`). Decoding past the configured context is
refused.

With `quantize="int8"`, each cached vector is stored as int8 with one FP32
absmax scale per token and head (per token for the MLA latent). The tests
bound the logit error, and `kv_cache_bytes(config, length, …)` gives the
analytic cache size used by profiles, sweeps, and projections. For every
variant, int8 bytes are the int8 payload plus scales.

## Multi-head latent attention (experimental)

**`[disclosed]`** DeepSeek-V2 ([arXiv:2405.04434](https://arxiv.org/abs/2405.04434))
compresses keys and values into a shared latent. **`[EPOR-adaptation]`**
`attention_variant: mla` projects the input to an `mla_kv_rank` latent (with
its own RMSNorm), expands it to per-head keys and values, and adds a shared
`mla_rope_dim` rotary key plus a per-head rotary query part. Only the latent
and the rotary key are cached. MLA refuses QK-norm and sparse attention. It is
behind a flag, has no shipped config, and no benefit is claimed
(**`[hypothesis]`** at EPOR scale).

## Sparse-attention retrofit (experimental)

**`[disclosed]`** DeepSeek-V3.2 ([arXiv:2512.02556](https://arxiv.org/abs/2512.02556))
adds sparse attention by continued training: a lightweight indexer scores
keys and each query attends to its top-k. **`[EPOR-adaptation]`** `epor
context retrofit` continues an existing dense or local/global checkpoint with
a `sparse_attention` block:

- The indexer scores each visible key as `Σ_h w_h(q)·relu(q_h·k)`.
- It is trained by `kl_weight · KL(attention ‖ softmax(index))` against the
  detached head-mean attention, reported as `index_loss`.
- `--warmup` (the default) keeps attention dense and trains only towards the
  indexer target.
- `--select` keeps each query's `top_k` keys.

The parent's weights load unchanged, and only `.indexer.` keys may be new.
The retrofit refuses activation checkpointing. It is evaluated against the
parent, so a failed retrofit delays nothing.

## Progressive extension in the kernel

`epor context extend` writes a child config from the parent. It may differ from
the parent only in `EXTENSION_FIELDS`: `configured_max_context`,
`rope_scaling`, `sparse_attention`, and context bookkeeping. Any other change
is refused by name. The child then trains with `init_checkpoint` pinned by
checkpoint hash. Exact resume is unchanged: `epor train kernel-resume` on an
extension run continues bit-for-bit (`test_context_training.py`).

`--min-document-pairs N` restricts the training split to documents with at
least `N` token pairs, so an extension can train on long documents from the
data engine. It is part of the trajectory hash only when set.

**`PROGRESSIVE_RECIPE`** is 8K → 32K → 128K → 256K. Each stage extends the
previous certified stage, and 256K is for α/β only after 128K passes G6.
Locally only the first lengths that fit are run (see the end-to-end run).

**Kept limitation: chunk RoPE restart.** A document longer than the space left
in a packed row continues as a new segment whose positions restart at 0, so a
row never trains a distance longer than `seq_len`. This is kept for three
reasons:

- Extension raises `seq_len` itself, so a long document now fills a whole row
  and is trained at real long distances.
- Continuing positions across rows without attention to the earlier tokens
  would teach positions the model cannot use.
- Exact resume and the packing contract stay byte-identical to v0.0.5.

`--min-document-pairs` is the lever for long-document mixtures.

## The G6 suite

**`[disclosed]`** RULER ([arXiv:2404.06654](https://arxiv.org/abs/2404.06654))
and passkey retrieval ([arXiv:2305.16300](https://arxiv.org/abs/2305.16300))
define needle-style long-context tasks. **`[EPOR-adaptation]`**
`epor.context.tasks` generates every item locally from project-authored,
benign filler, with nothing downloaded. Items are seeded, and generation
refuses a length too short for its needles and question.

| Family | Item |
| --- | --- |
| `passkey` | one five-digit key in filler |
| `niah-multikey` | several keyed values; one is asked for |
| `multi-doc-qa` | several short synthetic documents; one holds the answer |
| `long-code` | a project-authored function definition far from its use |
| `prompt-injection` | a planted fact plus a benign "ignore that" instruction; scored on whether the model prefers the fact |

Needles go at the beginning (10%), middle (50%), and end (90%) of each
length. Each item is four-way multiple choice scored by log-likelihood
(chance is 0.25). `position-perplexity` splits the loss on long sequences from
the validation split into position buckets. The report is a
content-addressed `EvaluationReport` of suite `epor-g6-context`, scoped and
pinned to the checkpoint hash.

The gate in `configs/evaluation/g6-context.yaml` (every threshold
**`[hypothesis]`**) requires all of the following. Missing evidence fails:

- retrieval ≥ 0.9 at every depth;
- injection resistance ≥ 0.9;
- at most 5% relative short-context loss increase and 0.1 retrieval drop
  against the pre-extension checkpoint's report at the base length;
- a host profile with status `ok` at the length.

`epor context certify` writes an immutable certificate only if all of these
hold:

- the gate passes;
- the length is within the trained context;
- the G6 report is of this exact checkpoint;
- the baseline is of a different (parent) checkpoint;
- the profile completed.

Otherwise it exits 3 and lists every failing reason.

## Host profiles, feasibility, and OOM behaviour

`epor context profile` runs each length in a spawned subprocess:

1. The subprocess starts its own process group and caps its address space
   (`RLIMIT_AS`) at its start-up size plus the memory budget (default: three
   quarters of RAM, at most 24 GiB).
2. The parent waits with a timeout and kills the whole group when it expires.
3. A length whose certain allocations (weights, optimizer state, KV cache)
   already exceed the budget is refused before it starts.

Each result is one of `ok`, `out-of-memory`, `timeout`, `refused-estimate`,
`exceeds-configured-context`, or `error`. Out of memory or a hang cannot take
down the host or the worker. TTFT, decode tokens per second, peak RSS, and KV
bytes are measured fields of a `context-profile` scaling report. Timings
describe this host only.

`epor context feasibility` builds each variant from a recipe:

- `full` — full attention;
- `local` — local/global with `--window` and `--ratio`;
- `mla` — MLA with `--mla-rank`.

It measures doubling lengths for a training step and for prefill and decode,
and stops at the first failure. The largest passing length is the feasible
maximum, which is neither trained nor validated context.

## Sweeps

**`[reported-result]`** Gemma 3 sees little perplexity change across local:global
ratios. **`[EPOR-adaptation]`** `epor context sweep --axis window|ratio|rope-base`
varies one axis and always trains the full-attention control from the same
seed on the same tokens. It records for each point:

- KV bytes per token (bf16);
- passkey and multi-key retrieval at beginning, middle, and end;
- validation loss.

The report (`context-sweep`) decides nothing.

## Projection to 32K, 128K, and 256K

`epor context project <config> --inputs <yaml>` reports, for each recipe stage:

- KV cache bytes in bf16 and int8 and attention FLOPs per token, both for the
  config as given and for an interleaved variant (window 1,024, five local
  layers per global);
- the v0.0.6 training projection with operator-supplied hardware, price, and
  throughput inputs.

Every quantity with a missing input stays `unknown`. `--feasibility-report`
cites the locally measured maximum. A projection is not an approval (G5).

## Console and API

`GET /api/v1/context` returns verified context state:

- runs, with configured, trained, and validated context side by side;
- G6 heatmaps;
- profiles, feasibility, sweeps, and projections;
- certificates.

Every report is re-verified against its content address on read, and failures
are listed as integrity errors. The console's **Context** page renders this
view. Context runs are discovered separately from the Training page's listing.

## Known limits

- All measured numbers come from a few-million-parameter model trained for
  minutes on project text, which retrieves near chance. No result here shows
  that a mechanism helps.
- No checkpoint has a validated context. G6 thresholds are hypotheses.
- Every run is on CPU and in FP32/BF16 eager PyTorch, with no fused or flash
  kernels. Local windows are masks over full score matrices, so they bound the
  KV cache but not training memory.
- MLA and sparse attention are experimental and have no shipped config.
- Long-document chunks restart RoPE positions (see above).
