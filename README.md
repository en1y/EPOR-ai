# EPOR AI

<!-- project-version:start -->
> [!IMPORTANT]
> **Current project version: `v0.0.7`**
>
> This identifies the code in this checkout; it does not imply a formal release.
> Follow implementation status in [`docs/ROADMAP.md`](docs/ROADMAP.md).
<!-- project-version:end -->

EPOR AI is a reproducible research platform for training **original-weight**
language models. Its public families are **EPOR-α** (`epor-alpha`), **EPOR-β**
(`epor-beta`), and **EPOR-γ** (`epor-gamma`). Released weights start from an
independent random initialization; external models may provide licensed
teacher signals or verified synthetic data, never reused weights.

The v0.0.x program is intentionally small. Through v0.0.7 it validates research
provenance, the executable safety covenant and local authorization boundary,
configuration, tiny-model correctness and exact resume, a provenance-first
offline data engine, an original byte-level BPE tokenizer with
content-addressed token shards, a CPU training kernel with exact resume
across single-process, DDP, and FSDP2 runs, an evaluation and scaling
laboratory, and a knowledge-transfer and alignment pipeline whose model
feedback stays a candidate until humans audit it, before any target-size run
is considered. The 30B/12B/8B
family descriptions are gated research destinations, not bundled checkpoints
or fabricated performance claims.

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

Who is asking is part of that decision rather than a separate check beside it.
Every mutation requires a verified identity — the project owner, or an
operator holding an expiring delegation that names declared actions
explicitly — and the actor is folded into the second principle's assessment, so
an unverified or out-of-scope request produces a Principle 2 conflict that the
ordering then refuses. Reading stays anonymous throughout. Work the covenant
will not decide creates no job and opens a sanitized escalation instead,
carrying only a digest of the request; an owner approval binds one actor, one
digest, and one covenant hash, expires in fifteen minutes, and is spent exactly
once. A refusal is never approvable.

What this is not: a harm detector. The resolver enforces ordering, citation,
justification, and fail-closed escalation over assessments supplied to it.
Through v0.0.7, those assessments are reviewed per-action declarations over a
closed allowlist, which is only sound because that action space is small and
fixed.
The authorization model is local and single-machine: it verifies a bearer
credential held on this machine, and cannot defend against a compromised local
account that can read that credential from disk. Classifiers, model-behavior
evaluations beyond the G10 harness (system level since v0.0.2, model level
before and after alignment since v0.0.7), and red teaming are later versions.

## Quick start

Python 3.11, [uv](https://docs.astral.sh/uv/), Node 22.22.x, npm 10.x, and Linux
`setsid` (normally provided by `util-linux`) are required. Run the local console
from the repository root.

Install the frozen Python and UI dependencies once after cloning or whenever a
lockfile changes:

```bash
uv sync --frozen --extra dev --extra research --extra data --extra train-cpu
cd ui
npm ci
cd ..
```

Start the API, worker, and browser console together:

```bash
./start-epor.sh
```

Mint the owner credential once, before the first mutation. It is printed a
single time and stored only as a SHA-256 digest:

```bash
uv run --frozen --no-sync epor auth bootstrap
```

Open the console at [http://127.0.0.1:5173](http://127.0.0.1:5173). Every page
is readable without signing in. To queue, cancel, or retry work, use **Sign in**
in the console header and paste that credential; it is exchanged for a 12-hour
session cookie and is not kept by the page. Every session ends when the API
restarts. The live control-API reference is available at
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

The v0.0.3 data path begins with a reviewed registration and a contained local
file. It never crawls or fetches an arbitrary URL:

```bash
mkdir -p .epor/import
cp configs/data/source-registration.example.yaml .epor/import/source-registration.yaml
# Replace every fixture value with reviewed evidence before ingestion.
uv run --frozen --no-sync epor data ingest \
  .epor/import/source-registration.yaml data/raw/source.txt
uv run --frozen --no-sync epor data build my-dataset-v1
uv run --frozen --no-sync epor data audit
```

See [`docs/DATA_ENGINE.md`](docs/DATA_ENGINE.md) for immutable layers, removal
tombstones, detector limitations, dataset cards, and control-job equivalents.

A build then feeds the v0.0.4 tokenizer, which trains on its train split only,
writes document-preserving token shards, and measures fertility on a held-out
split:

```bash
uv run --frozen --no-sync epor tokenizer train my-dataset-v1 <build-id>
uv run --frozen --no-sync epor tokenizer shard my-dataset-v1 <build-id> <tokenizer-id>
uv run --frozen --no-sync epor tokenizer evaluate <tokenizer-id> my-dataset-v1 <build-id>
```

Every tokenizer so far is proxy-scale; the 65,536-ID vocabulary is a declared
target, not a trained result. See [`docs/TOKENIZER.md`](docs/TOKENIZER.md) for
the reserved-ID table, artifact and shard formats, removal, and boundaries.

The v0.0.5 training kernel then trains a bounded dense model on those shards.
The shipped tiny recipe has a 260-ID debug vocabulary, so `--vocab-from-build`
trains an explicitly derived config with the build's vocabulary instead of
resizing the recipe:

```bash
uv run --frozen --no-sync epor train kernel configs/models/epor-tiny.yaml \
  --tokenized-id <tokenized-id> --output runs/kernel-tiny --max-steps 200 --vocab-from-build
uv run --frozen --no-sync epor train kernel-resume runs/kernel-tiny --max-steps 400
uv run --frozen --no-sync epor eval kernel runs/kernel-tiny
```

Add `--strategy ddp|fsdp2 --world-size 2` for a local loopback-only process
group. Ctrl-C checkpoints the last completed step and exits `130`; resume
continues the same lineage exactly. Every kernel result so far is a proxy-scale
local measurement, not a trained or validated model. See
[`docs/TRAINING.md`](docs/TRAINING.md) for settings, the exact-resume contract,
DDP/FSDP2 findings, and limitations.

The v0.0.6 laboratory evaluates and scales those runs. Suites score a
checkpoint and store a content-addressed report; code tasks run only with
`--allow-code-execution`, a separately authorized action, in a defense-in-depth
sandbox that is not a security boundary:

```bash
uv run --frozen --no-sync epor eval suite runs/kernel-tiny --suite fixtures
uv run --frozen --no-sync epor eval suite runs/kernel-tiny --suite safety
uv run --frozen --no-sync epor eval contamination --training-data <tokenized-id>
uv run --frozen --no-sync epor eval compare <baseline-report-id> <candidate-report-id>
uv run --frozen --no-sync epor scale ladder
uv run --frozen --no-sync epor scale profile configs/models/ladder/epor-ladder-10m.yaml
uv run --frozen --no-sync epor scale sweep --tokenized-id <tokenized-id> \
  --rung configs/models/epor-tiny.yaml --budget 8192 --budget 16384 --output runs/sweep
uv run --frozen --no-sync epor scale fit <sweep-report-id>
uv run --frozen --no-sync epor scale project --inputs <your-inputs.yaml> \
  --target configs/models/ladder/epor-ladder-1b.yaml
```

Only the 10M ladder rung runs on the reference machine; 100M, 300M, and 1B are
projection-only. Every result is a harness-fixture, proxy-scale, or host
measurement, and a projection provisions nothing and is never an approval. See
[`docs/EVALUATION.md`](docs/EVALUATION.md) and
[`docs/SCALING.md`](docs/SCALING.md).

The v0.0.7 alignment pipeline fine-tunes, distills, and preference-optimizes
those checkpoints on recorded, rights-checked data in the `epor-chat-v1`
format, cites covenant obligations by key for critique and revision, keeps AI
judge decisions and model-written evaluation items as candidates until a
verified human audits a sample of accepted and rejected cases, and measures
overoptimization and G10 behaviour before and after:

```bash
uv run --frozen --no-sync epor align fixtures
uv run --frozen --no-sync epor align dataset sft --checkpoint runs/kernel-tiny --include fixtures
uv run --frozen --no-sync epor train sft runs/kernel-tiny --dataset <sft-dataset-id> --output runs/sft
uv run --frozen --no-sync epor align critique runs/sft --example <record-id> --obligation no-harmful-output
uv run --frozen --no-sync epor align judge runs/sft --subject revision --all-unjudged
uv run --frozen --no-sync epor align audit sample <judging-run-id>
uv run --frozen --no-sync epor align audit verdict <item-id> accept
uv run --frozen --no-sync epor align dataset preference --checkpoint runs/sft --include fixtures --include ai-judge
uv run --frozen --no-sync epor train preference runs/sft --dataset <preference-dataset-id> --output runs/dpo
uv run --frozen --no-sync epor align monitor runs/dpo --reference runs/sft --dataset <preference-dataset-id>
uv run --frozen --no-sync epor align g10 --before runs/sft --after runs/dpo
```

Nothing is ever marked audited automatically, every monitor threshold is a
hypothesis, and a G10 regression fails closed. No result is evidence of
alignment or safety. See [`docs/ALIGNMENT.md`](docs/ALIGNMENT.md).

The tiny reference path can be checked independently:

```bash
uv run --frozen --no-sync epor doctor
uv run --frozen --no-sync epor config validate configs/models/epor-tiny.yaml
uv run --frozen --no-sync epor train pretrain configs/models/epor-tiny.yaml --output runs/smoke
uv run --frozen --no-sync pytest
```

Executable reference commands are deliberately capped at 50 million structural
parameters, a 1 MiB in-memory corpus, 1,000 training steps, 4,096 tokens per
batch, 10,000 evaluation batches, and a 768 MiB trusted-checkpoint container.
Their config, corpus, checkpoint, and output paths must resolve beneath the
source checkout (or the explicit `EPOR_PROJECT_ROOT`). These are v0.0.1
correctness limits, not configurable scale-training defaults.

The browser console is built separately from `ui/`; it calls only the
allowlisted `/api/v1` control API. Neither the API nor the UI can execute a
client-provided shell command, provision paid compute, or fetch an arbitrary
URL.

The console separates its work into thirteen pages: **Overview**, **Training**,
**Evaluation**, **Alignment**, **Scaling**, **Data engine**, **Tokenizer**, **Jobs**,
**Safety covenant**, **Access**, **Model families**, **Research ledger**, and
**Documentation**. The data page
shows reviewed source registrations, immutable-layer counts, quarantines,
removals, builds, and audit state without exposing document content. The
tokenizer page lists tokenizers with declared and learned vocabulary kept
apart, tokenized builds, and fertility reports with their scope and any
in-sample or round-trip warning. The training page lists verified
training-kernel runs, including nested sweep, mixture, and alignment outputs
under their location, with their losses, MTP losses, throughput, precision,
topology, features, checkpoint, and validation result, labelled as proxy-scale
measurements, beside the training and evaluation jobs. The evaluation page
shows verified suite reports with scoped metrics, calibration, contamination,
memorization, G10 scores under their claim note, and comparisons; the scaling
page shows the ladder, profiles, sweeps, mixtures, fits, and projections with
their unknown inputs and approval requirement, and has no launch control. The
alignment page shows provenance record counts, the human audit queue with
verdict controls for an identity holding the `align_audit` scope, judge
calibration and admission, overoptimization monitors, and the G10 comparison
before and after alignment. The documentation reader serves
tracked first-party Markdown only. Ignored
third-party research downloads are never rendered as console content; the
research ledger links those sources at their canonical URL instead.

Version 0.0.7 is operated from a source checkout. The wheel and source archive
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
CPU-only wheel index; the `eval` extra uses that same CPU backend and adds
SymPy for math checking (with lm-eval and tree-sitter pinned for later suites),
so the evaluation harness works without another profile.

## Project map

- [`docs/ROADMAP.md`](docs/ROADMAP.md) — release ladder and acceptance gates.
- [`docs/README.md`](docs/README.md) — documentation index.
- [`research/catalog.yaml`](research/catalog.yaml) — canonical source ledger;
  downloaded and extracted content remains ignored.
- `src/epor/data/` — rights registration, immutable layers, policy, builds, and audit.
- `src/epor/tokenizer/` — reserved-ID spec, BPE training, codec, token shards,
  and fertility reports.
- `src/epor/models/` and `src/epor/training/` — canonical PyTorch reference and
  the v0.0.5 training kernel.
- `src/epor/evaluation/` and `src/epor/scaling/` — the v0.0.6 evaluation harness,
  code sandbox, and scaling laboratory.
- `src/epor/alignment/` — the v0.0.7 chat format, teacher-data provenance,
  datasets, DPO, covenant-cited critique, judging, audits, monitors, and G10.
- `src/epor/control/` — loopback API, durable job index, and safe worker.
- `ui/` — React/TypeScript local management console.

Project code is proprietary and all rights are reserved. Model-weight terms
are assigned only after separate provenance audits, and nothing here grants
any rights to third-party research, datasets, or future weights.
