# Documentation

- [Roadmap](ROADMAP.md) — model-family destinations, version ladder, public
  interfaces, contracts, and acceptance gates.
- [Architecture](ARCHITECTURE.md) — model, artifact, research, training, and
  local-control boundaries.
- [Data governance](DATA_GOVERNANCE.md) — rights, lineage, privacy, filtering,
  removals, teacher data, and release requirements.
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

No target-size model has been trained or released by v0.0.1. Parameter counts,
context lengths, and architecture descriptions are plans until their named gates
produce measured artifacts.
