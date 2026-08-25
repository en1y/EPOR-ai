# Contributing

EPOR is pre-alpha research infrastructure. Prefer small changes that preserve
provenance, reproducibility, and local safety over speculative scale features.

## Development workflow

```bash
uv sync --frozen --extra dev --extra research --extra train-cpu
uv run pytest
uv run ruff check .
uv run pyright
```

Build the browser console with the pinned Node and npm versions declared in
`.nvmrc` and `ui/package.json`:

```bash
cd ui
npm ci
npm run build
```

Use Python 3.11 through `uv run`. Tests must not require the network, paid APIs,
special accelerators, or an existing research cache. Keep generated corpora tiny
and deterministic.

Do not commit raw/extracted papers, datasets, checkpoints, model weights, GGUF
files, runtime databases, logs, secrets, or build output. Add sources through the
tracked catalog and write a paraphrased, evidence-classified dossier instead.

## Change requirements

- Preserve public display names and machine IDs.
- Version schema changes and include migration/compatibility notes.
- Record seeds and hashes for generated artifacts.
- Add negative tests for URL, path, state-transition, and deserialization
  boundaries.
- Never invent metrics or promote configured context to validated context.
- Do not add an external model, dataset, or service without license/terms,
  provenance, privacy, and cost analysis.
- Do not initialize EPOR from third-party weights.

Research claims use the labels in [research/README.md](research/README.md).
Architecture changes tied to a paper must include a compute-matched baseline and
an explicit EPOR adaptation; reported third-party results are not EPOR results.

Pull requests should state scope, user-visible behavior, tests run, artifact or
schema compatibility, security/provenance impact, and remaining limitations.
