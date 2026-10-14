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
refuses a length too short for its needles, question, and answer;
an item at length `L` occupies at most `L` scored positions, so the full
scoring window, and therefore the trained context itself, can be evaluated.

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
maximum, which is neither trained nor validated context. The training probe
is one forward and backward pass on the kernel's segment-masked attention
path, with no optimizer, data pipeline, or validation pass, so it is a lower
bound on a kernel step. Pass the vocabulary the run will really train with
(`--vocab-size`): at 65,536 IDs the FP32 logits of one 8,192-token row alone
take 2 GiB.

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

## End-to-end run (local, proxy scale)

The run used the CPU reference machine (12 cores, 31 GiB RAM, no GPU) while
other desktop applications were running. The corpus was 130 project-authored
documentation and source files ingested as twelve groups:

- dataset split 76/22/32 documents for train/validation/test;
- a 9,829-entry tokenizer and 396,236 tokens of shards.

**Feasible maximum per variant.** These were measured on `epor-reference`
with the trained model's 65,536-entry vocabulary and a 6 GiB budget per
measurement:

| Variant | Training step fits up to | Prefill + decode fits up to | First failure (train / inference) |
| --- | --- | --- | --- |
| full attention (GQA) | 2,048 (3.01 GiB peak) | 16,384 (4.69 GiB, TTFT 15.9 s) | out-of-memory at 4,096 / refused-estimate at 32,768 |
| local/global (window 1,024, 3:1) | 2,048 (3.02 GiB) | 16,384 (4.58 GiB, TTFT 15.0 s) | out-of-memory at 4,096 / refused-estimate at 32,768 |
| MLA (rank 32) | 2,048 (3.44 GiB) | 8,192 (4.23 GiB, TTFT 12.0 s) | out-of-memory at 4,096 / out-of-memory at 16,384 |

Every failure ended cleanly inside its capped subprocess. A `refused-estimate`
was refused before it ran, because its logits alone exceed the budget.

- With a 9,829-entry vocabulary the same search reached 8,192 for training
  (4,096 for MLA). At this scale the output layer, not attention, sets the
  training ceiling.
- Local windows did not raise any maximum, because the window is a mask over a
  full score matrix.
- Eager MLA expands its latent to full keys and values, so it costs more
  memory.

**Base, extension, and G6.**

1. The base model, `epor-reference` with 35.5M parameters at the 65,536-entry
   vocabulary, trained 300 steps at 512 tokens.
2. `epor context extend` to 8,192 failed in its first step. Under a soft
   9 GiB address-space cap this was a clean 2 GiB logits allocation error, at
   6.6 GB peak RSS. An earlier attempt without a cap was stopped by the host's
   OOM killer at 9.2 GB.
3. Extension to 4,096 with YaRN (factor 8 from 512) trained 30 steps on the
   38 documents with at least 2,048 token pairs (`--min-document-pairs`):
   - about 904 tokens/s;
   - peak RSS 6.65 GiB, including its validation pass, which is above the
     6 GiB probe budget;
   - trained context 4,096, validated context 0.
4. G6 at 4,096 with the default scoring batch of four failed cleanly on a
   4.3 GB allocation, the logits of four rows. With `--batch-size 1` it
   completed.

The locally feasible kernel training length was therefore 4,096 with this
vocabulary on this host.

Retrieval is at or near chance everywhere, with two items per cell.
`certify` at 4,096 exited 3 with thirteen failing reasons: every retrieval
depth below 0.9, and injection resistance 0.5. Short-context regression and
the host profile passed.

**KV cache** (4,096-token checkpoint):

| Length | FP32 cache | int8 cache | TTFT FP32 / int8 | Decode tok/s FP32 / int8 |
| --- | --- | --- | --- | --- |
| 512 | 3.0 MiB | 0.8 MiB | 0.19 / 0.19 s | 73 / 67 |
| 2,048 | 12.0 MiB | 3.2 MiB | 0.80 / 0.81 s | 39 / 59 |
| 4,096 | 24.0 MiB | 6.4 MiB | 1.77 / 1.97 s | 26 / 39 |
| 8,192 | — | — | refused as `exceeds-configured-context` | — |

The int8 cache holds 26.6% of the FP32 bytes. With a 1,536-token prefill and
512 decode steps, it moved logits by at most 0.175 (mean 0.0042), and the
argmax agreed on 99.8% of positions. The FP32 cache stayed within 4.8e-5 of
the no-cache forward. Decode timings are single short runs on a shared host.

**Sparse retrofit** (512 tokens, 10 warm-up and 10 selection steps, top-k 64)
changed the validation main loss from 5.747 (base) to 5.748 after warm-up
and 6.247 after selection, with near-chance G6. That is a regression, as
expected for a 10-step retrofit, and it is not a result about sparse
attention.

**Sweeps** (40 steps each, retrieval at 256 tokens): every point retrieved
at the same near-chance level as its control, so they discriminate only on
cache size and loss:

| Sweep | KV bytes/token (bf16) | Validation loss |
| --- | --- | --- |
| window, tiny | 256 control; 144 (w=32); 192 (w=128) | 7.296 / 7.288 / 7.302 |
| ratio, `epor-reference`, w=64 | 3,072 control; 1,920 (1:1); 1,536 (2:1); 1,152 (5:1) | 6.969 / 6.986 / 7.009 / 7.012 |
| split RoPE base, tiny | 256 control; 160 shared; 160 (local base 1,000) | 7.296 / 7.289 / 7.292 |

**Projection for `epor-beta`** (KV cache per sequence, bf16 / int8):

| Stage | As configured | Interleaved local/global | Note |
| --- | --- | --- | --- |
| 32K | 5.0 / 2.58 GiB | 0.88 / 0.46 GiB | progressive stage |
| 128K | 20.0 / 10.31 GiB | 3.13 / 1.62 GiB | progressive stage |
| 256K | 40.0 / 20.62 GiB | 6.13 / 3.16 GiB | α/β only, after 128K passes G6 |

Time, device-hours, cost, and energy stay `unknown` because the example
inputs leave hardware and prices empty.

## Known limits

- All measured numbers come from a few-million-parameter model trained for
  minutes on project text, which retrieves near chance. No result here shows
  that a mechanism helps.
- No checkpoint has a validated context. G6 thresholds are hypotheses.
- Every run is on CPU and in FP32/BF16 eager PyTorch, with no fused or flash
  kernels. Local windows are masks over full score matrices, so they bound the
  KV cache but not training memory.
- MLA and sparse attention are experimental and have no shipped config.
- `epor context extend` and `epor train kernel` have no memory guard of
  their own, so a kernel step above the host's memory can reach the OOM
  killer. Size an extension with `epor context feasibility` at the real
  vocabulary, and run it under an address-space limit (`ulimit -S -v`).
  Score G6 at long lengths with `--batch-size 1`.
- Long-document chunks restart RoPE positions (see above).
