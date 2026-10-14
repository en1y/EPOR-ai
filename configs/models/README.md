# Model configurations

These five files are executable architecture contracts, not released checkpoints
or benchmark claims:

- `epor-tiny.yaml`: roughly 115K parameters for fast CI only.
- `epor-reference.yaml`: canonical 10.4M-parameter pure-PyTorch reference.
- `epor-alpha.yaml`: planned EPOR-α target configuration.
- `epor-beta.yaml`: planned EPOR-β target configuration.
- `epor-gamma.yaml`: planned EPOR-γ target configuration.

The four context fields have deliberately different meanings:

- `configured_max_context`: construction/configuration ceiling.
- `trained_max_context`: longest sequence used in training.
- `validated_max_context`: longest sequence that passed the context suite.
  Since v0.0.8 measured trained context lives in run manifests and validated
  context only in G6 certificates (`epor context certify`); neither is ever
  written back into these files.
- `operational_default_context`: conservative default supported by evidence.

The public-family files therefore start with zero trained, validated, and
operational context. Their hardware profiles are marked `planned`. Parameter
counts are implementation targets checked allocation-free on PyTorch's meta
device. EPOR-γ's declared core is a research target; the v0.0.1 dense reference
graph counts the full model as active until elastic extraction is implemented.

## Optional architecture features

These fields default to the original single-token, pre-norm control block. At
their defaults they are left out of the canonical config JSON, so adding them
did not change the hash of any existing configuration or run manifest; a
non-default value changes the computation and therefore the hash.

- `qk_norm` (default `false`): per-head RMSNorm on queries and keys after
  projection and before RoPE, as described in the Gemma 3 report
  (`gemma-3-technical-report`, `disclosed`). The `false` block is the control.
- `mtp_depth` (default `0`): number of sequential DeepSeek-V3-style
  multi-token-prediction modules (`deepseek-v3`, `disclosed`). Each depth adds
  a projection and one decoder block that share the embedding and output head.
- `mtp_loss_weight` (default `0.3`, DeepSeek-V3's reported early-training
  value): weight on the mean MTP loss. It must stay at its default while
  `mtp_depth` is `0`.

None of the shipped files enables them. Whether QK-norm or MTP is worth its cost
at EPOR scale is undecided (`hypothesis`) until G3/G5 compares each against its
control at matched tokens; the parameter report lists MTP's extra parameters
and per-token FP32 logit bytes so that memory cost is visible. Activation
checkpointing is a runtime option of the decoder, not a config field, because it
does not change the model's math.

## Context fields (v0.0.8)

These fields are hash-neutral at their defaults like the features above. None
of the shipped files sets them; `epor context extend` and the sweeps derive
configs that do. See [`docs/CONTEXT.md`](../../docs/CONTEXT.md).

- `rope_scaling` (default none): `linear`, `ntk`, or `yarn` with `factor` and
  `original_max_context` (YaRN also takes `beta_fast` 32, `beta_slow` 1, and
  `attention_factor`, default `0.1·ln(factor) + 1`). It scales only the
  layers that use `rope_theta`.
- `local_rope_theta` (default none): a separate RoPE base for local layers
  (Gemma 3's split base, `disclosed`).
- `sliding_window` and `local_layers_per_global` (default `0`, set together):
  in every group of `local_layers_per_global + 1` layers, the first attend
  only to the previous `sliding_window` tokens and the last attends globally.
- `attention_variant` (default `gqa`), `mla_kv_rank`, `mla_rope_dim`:
  experimental multi-head latent attention (`deepseek-v2`, `disclosed`). It
  caches only the latent and a shared positional key.
- `sparse_attention` (default none): the experimental DeepSeek-V3.2-style
  indexer and top-k retrofit (`deepseek-v3-2`, `disclosed`). It is added only
  to an existing checkpoint by `epor context retrofit`.

No benefit of any of these at EPOR scale is claimed (`hypothesis`).

## Training-kernel runs

`epor-tiny.yaml` keeps the 260-ID vocabulary of the v0.0.1 debug byte
tokenizer, which the CI smoke path uses. The v0.0.5 training kernel needs the
tokenized build's vocabulary instead. These files are never edited for that:
`epor train kernel --vocab-from-build` (or `vocab_from_build` in a job) trains a
derived config whose `vocab_size` and declared parameter counts come from the
build, and `--qk-norm`/`--mtp-depth` derive feature variants the same way. A
derived config has its own canonical hash, recorded in the run manifest and in
every checkpoint; the hash of the file it came from does not change. See
[`docs/TRAINING.md`](../../docs/TRAINING.md).

## Scaling ladder

`ladder/` holds the v0.0.6 scaling-laboratory rungs. They are architecture
contracts for sweeps and projections, not trained or validated models, and
adding them changed no shipped config hash.

| File | Non-embedding parameters | Structural total (65,536 vocab) | Local execution |
|------|--------------------------|---------------------------------|-----------------|
| `ladder/epor-ladder-10m.yaml` | 10,326,912 | 35,492,736 | local (within the 50M kernel bound) |
| `ladder/epor-ladder-100m.yaml` | 103,243,008 | 153,574,656 | projection only |
| `ladder/epor-ladder-300m.yaml` | 301,517,824 | 368,626,688 | projection only |
| `ladder/epor-ladder-1b.yaml` | 1,013,012,480 | 1,147,230,208 | projection only |

Rungs are sized by non-embedding parameters (the `N` of Kaplan et al. 2020),
because at the 65,536-entry target vocabulary the tied embedding alone would
dominate a total-count ladder; each lands within 5% of its nominal size. Shapes
follow the shipped recipes: `d_ff = 3 * d_model`, grouped-query attention, 64-wide
heads up to 300M and 128-wide at 1B. Local sweeps train a derived config with
the tokenized build's vocabulary, the same derivation `--vocab-from-build` uses;
the rung file is never edited. See [`docs/SCALING.md`](../../docs/SCALING.md).
