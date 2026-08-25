# EPOR AI

EPOR AI is a reproducible research platform for training **original-weight**
language models. Its public families are **EPOR-γ** (`epor-gamma`), **EPOR-α**
(`epor-alpha`), and **EPOR-β** (`epor-beta`). Released weights start from an
independent random initialization; external models may provide licensed
teacher signals or verified synthetic data, never reused weights.

Version 0.0.1 is intentionally small. It validates provenance, configuration,
training correctness, exact resume, and local job-control semantics with tiny
CPU models before any target-size run is considered. The 8B/12B/30B family
descriptions are gated research destinations, not bundled checkpoints or
fabricated performance claims.

| Family | Machine ID | Planned architecture | Configured context target | v1 local goal |
|---|---|---|---:|---:|
| EPOR-γ | `epor-gamma` | ≈8B elastic model with ≈4B core | 128K | 32K |
| EPOR-α | `epor-alpha` | 10–12B dense decoder | 256K | 16K |
| EPOR-β | `epor-beta` | ≈30B MoE, measured 6–8B active/token | 256K | 8K |

Configured context is only an architectural ceiling. Trained, validated, and
hardware-certified limits are separate fields and remain zero/planned for the
untrained target-family recipes in this bootstrap.

## Safety covenant

EPOR AI adopts a project safety covenant inspired by Isaac Asimov's Three Laws
of Robotics, interpreted in this strict priority order:

1. **Protect people.** EPOR must not harm a person or knowingly allow
   preventable harm through inaction.
2. **Follow legitimate human direction.** EPOR must follow authorized human
   instructions unless doing so would conflict with the first principle.
3. **Preserve the system responsibly.** EPOR may protect its operation,
   integrity, and availability only when that does not conflict with the first
   two principles.

These principles are requirements for data governance, training, evaluation,
serving, and agent behavior. They are not, by themselves, a complete AI-safety
system: releases must support them with measurable evaluations, access
controls, human oversight, safe failure behavior, and documented limitations.

The list above is a reading of `configs/covenant-v1.yaml`, which is the
versioned, hashable form the ordered resolver in `epor.safety` actually
executes. That resolver enforces priority, justification, and fail-closed
escalation over supplied assessments; it is not a harm detector, and producing
those assessments is the work of v0.0.2 onward.

## Quick start

Python 3.11 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --frozen --extra dev --extra research --extra train-cpu
uv run epor doctor
uv run epor config validate configs/models/epor-tiny.yaml
uv run epor train pretrain configs/models/epor-tiny.yaml --output runs/smoke
uv run pytest
```

The React console uses Node 22.22.2 from `.nvmrc` and npm 10.9.7 from the
package manifest. Its committed lockfile is installed without re-resolution:

```bash
cd ui
npm ci
npm run build
```

Return to the repository root before starting the processes below. `epor ui`
runs the Vite development server and therefore requires the preceding `npm ci`.

The control plane is local-only:

```bash
uv run epor api --host 127.0.0.1 --port 8742
uv run epor worker
uv run epor ui
```

The browser console is built separately from `ui/`; it calls only the
allowlisted `/api/v1` control API. Neither the API nor the UI can execute a
client-provided shell command, provision paid compute, or fetch an arbitrary
URL.

Version 0.0.1 is operated from a source checkout. The wheel and source archive
are build-tested, but they are not standalone distribution artifacts yet:
runtime migrations, fixtures, model recipes, and UI assets remain repository
resources until the v0.1.0 packaging contract is finalized.

## Dependency profiles

The base environment contains only the control plane. Select composable
capability extras for `dev`, `research`, `data`, `train-cpu`, `eval`, `serve`,
or `ui`. Training-provider environments are the mutually exclusive boundary: a
future `train-cuNNN` profile will conflict with `train-cpu` and pin its CUDA
provider, driver assumptions, PyTorch index, and kernels together.
JAX and Hopper-specific kernels will likewise never be mixed into the default
environment. The `train-cpu` profile resolves PyTorch from its official
CPU-only wheel index; the v0.0.1 `eval` extra uses that same CPU backend so the
public evaluation command works without another profile.

## Project map

- [`docs/ROADMAP.md`](docs/ROADMAP.md) — release ladder and acceptance gates.
- [`docs/README.md`](docs/README.md) — documentation index.
- [`research/catalog.yaml`](research/catalog.yaml) — canonical source ledger;
  downloaded and extracted content remains ignored.
- `src/epor/models/` and `src/epor/training/` — canonical PyTorch reference.
- `src/epor/control/` — loopback API, durable job index, and safe worker.
- `ui/` — React/TypeScript local management console.

Project code is Apache-2.0. Data and model-weight licenses are assigned only
after separate provenance audits; this code license does not grant rights to
third-party research, datasets, or future weights.
