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
