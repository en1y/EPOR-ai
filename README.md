# EPOR AI

EPOR AI is a reproducible research platform for training **original-weight**
language models. Its public families are **EPOR-α** (`epor-alpha`), **EPOR-β**
(`epor-beta`), and **EPOR-γ** (`epor-gamma`). Released weights start from an
independent random initialization; external models may provide licensed
teacher signals or verified synthetic data, never reused weights.

Version 0.0.1 is intentionally small. It validates provenance, configuration,
training correctness, exact resume, and local job-control semantics with tiny
CPU models before any target-size run is considered. The 30B/12B/8B family
descriptions are gated research destinations, not bundled checkpoints or
fabricated performance claims.

Families are listed alphabetically, and intended capability descends in the same
order: α is the flagship, γ the compact local model.

| Family | Machine ID | Planned architecture | Configured context target | v1 local goal |
|---|---|---|---:|---:|
| EPOR-α | `epor-alpha` | ≈30B MoE, measured 6–8B active/token | 256K | 8K |
| EPOR-β | `epor-beta` | 10–12B dense decoder | 256K | 16K |
| EPOR-γ | `epor-gamma` | ≈8B elastic model with ≈4B core | 128K | 32K |

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
executes. Each principle is stated there as an operative law with numbered
obligations, and any decision that claims a principle is engaged must cite the
obligation by key.

The covenant is enforced, not advisory. `epor.actions` declares the covenant
standing of every action EPOR can execute, and both entry points resolve
against it: the control plane admits a job only after the covenant permits it —
before its specification is parsed — writing the resolution and covenant hash
into the append-only audit record, with the worker re-resolving at dispatch;
and every CLI command that executes work authorizes first and exits non-zero if
refused. An action with no reviewed declaration escalates instead of running,
so work cannot become executable by being forgotten. Cancellation and shutdown
are never gated, because the second principle outranks the third.

What this is not: a harm detector. The resolver enforces ordering, citation,
justification, and fail-closed escalation over assessments supplied to it. In
v0.0.2 those assessments are reviewed per-job-type declarations over a closed
four-entry allowlist, which is only sound because that action space is small
and fixed. Classifiers, model-behavior evaluations, and red teaming are later
versions.

## Quick start

Python 3.11, [uv](https://docs.astral.sh/uv/), Node 22.22.x, npm 10.x, and Linux
`setsid` (normally provided by `util-linux`) are required. Run the local console
from the repository root.

Install the frozen Python and UI dependencies once after cloning or whenever a
lockfile changes:

```bash
uv sync --frozen --extra dev --extra research --extra train-cpu
cd ui
npm ci
cd ..
```

Start the API, worker, and browser console together:

```bash
./start-epor.sh
```

Open the console at [http://127.0.0.1:5173](http://127.0.0.1:5173). The live
control-API reference is available at
[http://127.0.0.1:8742/api/v1/docs](http://127.0.0.1:8742/api/v1/docs).

Press `Ctrl+C` in the launcher terminal to stop the API, worker, and UI
together.

To verify the UI bundle independently:

```bash
cd ui
npm run build
cd ..
```

To run the services manually instead, start each process from the repository
root in a separate terminal:

```bash
uv run --frozen --no-sync epor api --host 127.0.0.1 --port 8742
```

```bash
uv run --frozen --no-sync epor worker
```

```bash
uv run --frozen --no-sync epor ui
```

Stop each manually started process with `Ctrl+C` in its terminal.

The tiny reference path can be checked independently:

```bash
uv run --frozen --no-sync epor doctor
uv run --frozen --no-sync epor config validate configs/models/epor-tiny.yaml
uv run --frozen --no-sync epor train pretrain configs/models/epor-tiny.yaml --output runs/smoke
uv run --frozen --no-sync pytest
```

The browser console is built separately from `ui/`; it calls only the
allowlisted `/api/v1` control API. Neither the API nor the UI can execute a
client-provided shell command, provision paid compute, or fetch an arbitrary
URL.

The console separates its work into six pages: **Overview** for environment and
boundary facts, **Training** for reference training and evaluation runs with
their loss and checkpoints, **Jobs** for every other typed job, **Model
families** for configured architecture plans, **Research ledger** for tracked
sources and their local verification state, and **Documentation** for the
project's own Markdown rendered in place, including Mermaid diagrams. The
documentation reader serves tracked first-party Markdown only. Ignored
third-party research downloads are never rendered as console content; the
research ledger links those sources at their canonical URL instead.

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
