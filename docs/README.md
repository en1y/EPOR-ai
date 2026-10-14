# Documentation

- [Roadmap](ROADMAP.md) — model-family destinations, version ladder, public
  interfaces, contracts, and acceptance gates.
- [Architecture](ARCHITECTURE.md) — model, artifact, research, training, and
  local-control boundaries.
- [Data governance](DATA_GOVERNANCE.md) — rights, lineage, privacy, filtering,
  removals, teacher data, and release requirements.
- [Provenance-first data engine](DATA_ENGINE.md) — v0.0.3 registration,
  immutable layers, policy sequence, commands, evidence, and limitations.
- [Original tokenizer and token shards](TOKENIZER.md) — v0.0.4 reserved-ID
  layout, train-split-only training, shard and report formats, removal, and
  evidence boundaries.
- [Training kernel](TRAINING.md) — v0.0.5 packed batching, optimizer and
  precision, DDP/FSDP2 on CPU, checkpoints and the exact-resume contract, fault
  injection, operator surfaces, and limitations.
- [Evaluation harness](EVALUATION.md) — v0.0.6 suites, task types and scoring,
  calibration, registered benchmark sources, the code sandbox and its gaps,
  contamination, memorization, and the G10 safety suite's claim boundary.
- [Scaling laboratory](SCALING.md) — v0.0.6 ladder, FLOPs model, host profiles,
  sweeps, mixture ablations, fits, and projections mapped to the compute policy.
- [Knowledge transfer and alignment](ALIGNMENT.md) — v0.0.7 chat format,
  teacher-data provenance, SFT and distillation on the kernel, the DPO
  baseline, covenant-cited critique, audited AI judging, overoptimization
  monitors, and model-level G10.
- [Compute policy](COMPUTE_POLICY.md) — cheap-first experiments, approval gates,
  abort rules, and hardware profiles.
- [Contributing](CONTRIBUTING.md) — environment, tests, claim discipline, and
  change requirements.
- [Security](SECURITY.md) — reporting, supported surface, trust boundaries, and
  secret handling.
- [Research dossiers](research/README.md) — evidence taxonomy and paraphrased
  Anthropic, DeepSeek, Gemma 3/3n, and MatFormer notes.

The tracked research ledger is [`research/catalog.yaml`](../research/catalog.yaml).
Downloaded bytes, extracted text, and local indexes live under ignored
`research/raw/`, `research/extracted/`, and `research/cache/` directories.

No target-size model or target-scale tokenizer has been trained or released by
v0.0.7; kernel runs, fine-tunes, DPO runs, evaluations, and sweeps so far are
proxy-scale local measurements, no alignment or safety claim follows from
them, and no target-size run has been proposed or approved. Parameter counts,
context lengths, vocabulary targets, and architecture descriptions are plans
until their named gates produce measured artifacts.
