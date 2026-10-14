# Scaling laboratory

v0.0.6 adds the scaling laboratory (`epor.scaling`): a 10M → 100M → 300M → 1B
model-size ladder, an analytic FLOPs model, measured host profiles, resumable
sweeps, data-mixture ablations, deterministic scaling-law fits, and compute,
storage, energy, and cost projections for a target run.

**Claim boundary.** On the CPU reference machine only the 10M rung can run;
100M, 300M, and 1B are **projection-only** and have never been allocated.
Sweep and mixture losses are `proxy-scale` measurements of small local models
on a named build; profiles are `host-measurement`s of this machine only; a
projection is arithmetic over operator-supplied inputs. No scaling claim about
a release-scale model follows from any of them, nothing here provisions
compute, and no target-size run has been proposed or approved.

## Commands and surfaces

```text
epor scale ladder [--vocab-size N]
epor scale profile CONFIG [--vocab-size N | --vocab-from-build TOKENIZED_ID]
                   [--batch-size --seq-len --warmup-steps --timed-steps --precision
                    --seed --threads --peak-flops FLOP/S --device-label]
epor scale sweep --tokenized-id ID --budget N... --output DIR [--rung CONFIG]...
                 [--seed N]... [training settings]
epor scale mixture CONFIG --tokenized-id ID --mixture ID:domain=w,...[@seed]...
                   --max-steps N --output DIR [--evaluation-split] [training settings]
epor scale fit SWEEP_REPORT_ID
epor scale project --inputs PATH --target CONFIG [--fit FIT_REPORT_ID]
epor scale reports [REPORT_ID]
```

Paths resolve beneath `EPOR_PROJECT_ROOT`; output is JSON with progress on
stderr; exit codes match [`EVALUATION.md`](EVALUATION.md) (`2` input or
verification, `3` covenant refusal, `130` cooperative stop). Rerunning a sweep
or mixture into the same `--output` resumes it.

| Surface | Action | Control job |
| --- | --- | --- |
| `scale profile` | `scale_profile` | `scale_profile` |
| `scale sweep` | `scale_sweep` | `scale_sweep` (at most 64 points) |
| `scale mixture` | `scale_mixture` | `scale_mixture` |
| `scale fit` | `scale_fit` | CLI only |
| `scale project` | `scale_project` | CLI only |
| `scale ladder`, `scale reports` | `report_review` | read-only endpoints |

Fits and projections stay on the CLI: they are cheap arithmetic over a stored
report and an operator-written file. Model configs for jobs must live under
`configs/models/`, and job outputs go to `<artifact_root>/<job id>/`. The
console's Scaling page reads `GET /api/v1/scaling/ladder`,
`GET /api/v1/scaling/reports`, and `GET /api/v1/scaling/reports/{id}`; it can
queue profiles, sweeps, and mixture ablations and has no launch, provision, or
approval control.

## Ladder

`configs/models/ladder/` holds four dense rungs sized by **non-embedding**
parameters (Kaplan's `N`), because at the 65,536-entry target vocabulary the
tied embedding alone is 25M-134M parameters and a total-count ladder would
mostly measure the vocabulary. Shapes follow the shipped recipes (64-wide
heads to 300M, 128 at 1B, `d_ff = 3 d_model`, grouped-query attention). The
files are architecture contracts, not trained models.

| Rung | Non-embedding | Structural total (65,536 vocab) | Reference machine |
| --- | ---: | ---: | --- |
| `epor-ladder-10m` | 10.33M | 35.49M | local |
| `epor-ladder-100m` | 103.24M | 153.57M | projection-only |
| `epor-ladder-300m` | 301.52M | 368.63M | projection-only |
| `epor-ladder-1b` | 1.013B | 1.147B | projection-only |

Local execution is decided by the training kernel's existing bound
(`validate_kernel_model`: dense, at most 50M structural parameters), so a rung
above it is recorded with its counts and a reason and never allocated. Local
runs train a derived config with the build's vocabulary, as
`--vocab-from-build` does; the rung file's hash never changes.

## FLOPs model

Per token, forward ≈ `2 N_matmul + 2 n_layer n_ctx d_attn` and training ≈
3 × forward (Kaplan et al. 2020, `disclosed`), which for long runs is
`C ≈ 6 N D`. EPOR's `N_matmul` includes the tied output-head projection, a
large share of small-model compute at a 65,536 vocabulary; a mixture-of-experts
layer counts only the experts a token visits; each MTP depth adds its
projection, block, attention term, and logit projection. These are
`EPOR-adaptation`s. Like Kaplan's formula the attention term is not halved for
causal masking, a slight over-count.

## Host profiles

`scale profile` builds the config on the CPU with deterministic algorithms and
runs untimed warmup steps then timed steps (forward, backward, clipping, AdamW)
on seeded synthetic tokens, at most 1,000 steps. The report's content holds the
config, batch geometry, precision, seeds, threads, and the analytic FLOPs; its
`measured` section holds tokens per second, per-step seconds, achieved FLOP/s
(analytic FLOPs per token × measured rate), process peak RSS, RSS after the
run, and the host snapshot. **MFU is `unknown` unless the operator supplies the
device peak** with `--peak-flops`; EPOR never looks it up. Synthetic tokens
measure compute only.

Host measurements on the reference machine (12 logical CPUs, 31 GiB, PyTorch
using 6 threads, FP32), from the v0.0.6 end-to-end run — measurements of this
host only, not predictions for any accelerator:

| Config | Batch × sequence | Tokens / s | Step (s) | Analytic FLOP/s | Peak RSS | MFU |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `epor-tiny`, 2,048-entry build vocabulary | 8 × 64 | ≈31,800 | 0.016 | ≈4.5 × 10¹⁰ | 0.44 GB | unknown |
| `epor-ladder-10m`, 65,536 vocabulary | 2 × 128 | ≈328 | 0.78 | ≈7.0 × 10¹⁰ | 1.54 GB | unknown |

## Sweeps

`scale sweep` expands rungs × token budgets × seeds into points. A local rung
trains through the kernel for `ceil(budget / tokens_per_batch)` steps with its
seed as both model and data seed and a complete cosine schedule of that length
(warmup capped at it), keeping one checkpoint. Each point's ID hashes the
derived config and trajectory; completed points are written under
`<output>/points/` and skipped on rerun, interrupted points resume from their
last checkpoint, and failed points are kept with their error because the
compute policy requires failed runs in scaling analysis. Losses, tokens, and
loss curves are report content; wall time, throughput, RSS, and run
directories are `measured`. Each attempt
(`<output>/runs/<point_id>/attempt-<n>`) is an ordinary kernel run, so the
console's Training page lists it under that location.

## Data mixtures

A mixture names target token shares over data-engine domain labels. The
kernel's `train_mixture` setting selects a deterministic, non-repeating subset
of one build's training documents whose token shares match the weights (the
scarcest domain is taken whole; others are filled in hash order), so no build,
tokenizer, or shard is rewritten and every mixture shares one tokenizer and one
validation split. `scale mixture` trains the same model for the same steps on
each mixture — matched training tokens, with epochs over each subset recorded —
then measures validation loss on each domain's validation documents. The
overall figure is the target-token-weighted mean of the per-domain losses,
which equals the whole-split mean when the pass is uncapped; under
`--evaluation-max-steps` each domain is capped separately, so a capped pass can
no longer let the first domain in document order stand in for the split. This
materialization is an `EPOR-adaptation`; the roadmap's
55/25/15/5 proxy mixture is a starting point, and a mixture is chosen only by a
human from measured losses. Validation is never re-weighted.

`train_mixture` is accepted wherever `PackedTrainingSettings` is validated, but
the `kernel_train` job specification deliberately has no such field and rejects
it; domain-weighted training reaches the control plane only through
`scale_mixture`, whose report records the mixture and its selection.

## Fits

`scale fit` fits `L(N) = E + A N^-α` at the largest token budget,
`L(D) = E + B D^-β` at the largest model, and the joint
`L(N, D) = E + A N^-α + B D^-β` (Kaplan 2020; Hoffmann et al. 2022, both
`disclosed`) to a sweep's completed points, averaging seeds per `(N, D)`. For
fixed exponents the model is linear in `E` and the coefficients, so the fit
solves weighted least squares (weights `1/L²`) in closed form over an exponent
grid and refines the best cell, with `E ≥ 0` and positive coefficients — a
standard-library `EPOR-adaptation` of Hoffmann et al.'s Huber/L-BFGS procedure
that always gives the same parameters for the same points. A fit needs at least
three distinct values per axis (five varied points for the joint form);
otherwise it reports `insufficient-points`. Warnings are explicit when there
are no residual degrees of freedom and when the points span less than an order
of magnitude, which is the normal situation for a local sweep. Applying these
forms to EPOR is a `hypothesis`.

## Projections

`scale project` answers the target-proposal list in
[`COMPUTE_POLICY.md`](COMPUTE_POLICY.md) (approval ladder step 4) plus G5's
MFU, throughput, and loss-curve expectation, from an operator-written inputs
file (copy `configs/scaling/projection-inputs.example.yaml`, whose values are
all null) and a target config. Every quantity is an estimate with a value,
unit, evidence label, basis, and the inputs it is missing; a missing input
makes every dependent quantity `unknown`, listed in `unknown_inputs`. Nothing
is fetched: prices, peaks, MFU, and power come from the operator, with their
sources recorded.

| Policy item | Projection field |
| --- | --- |
| Tokens | `tokens.training_tokens` (operator budget, else `tokens_per_parameter` × active parameters) |
| GPU-hours | `gpu_hours`: training and evaluation device-hours, wall-clock hours, projected tokens/s |
| Provider SKU and count | `provider` (operator input, `unknown` when absent) |
| Storage and egress | `storage` (checkpoints at 12 bytes/parameter, retained copies, `uint16` shards), `egress` |
| Evaluation | `evaluation` (passes × tokens × forward FLOPs) |
| Checkpoint cadence | `checkpoint_cadence` (every hours, keep, written) |
| Failure reserve | `failure_reserve` (fraction, cost, device-hours with reserve) |
| Carbon and energy | `energy_carbon` (device power × PUE; carbon intensity) |
| Hard budget and abort envelope | `budget_abort_envelope` (max spend, device-hours, wall-clock, expected loss from a fit, mandatory abort conditions) |
| Memory | `memory` (16 bytes/parameter training state vs aggregate device memory) |

Two documented defaults apply only when their input is null: 20 tokens per
parameter (Hoffmann et al.'s reported compute-optimal ratio, `disclosed`
there, applied to EPOR as a `hypothesis`) and a 0.25 failure-reserve fraction
(`EPOR-adaptation`). Every projection carries `approval.required: true` and the
statement that a projection is not authorization: a named human must approve
the exact immutable proposal (G5), and a material change needs a new approval.
The v0.0.6 end-to-end run exercised the arithmetic with placeholder inputs in a
scratch file; no projection with real quotes or measured pilot MFU exists.

## Reports

Every runner returns a sealed `ScalingReport`. Its `report_id` hashes the
content — kind, scope, claim boundary, harness, inputs, results, warnings,
citations — and excludes the `measured` section, so repeated measurements of
one experiment share an ID while every timing is kept. Each measurement is
stored once at `<data_root>/scaling-reports/<report_id>/<content_sha256>.json`
and both addresses are re-verified on load; edited files are reported as
integrity errors and never listed.

## Evidence and limits

Tests: `test_scaling_ladder.py`, `test_scaling_fit.py`,
`test_scaling_projection.py`, `test_scaling_runs.py`,
`test_training_mixture.py`, `test_laboratory_cli.py`, and
`test_laboratory_control.py`. Known limits: only the 10M rung runs locally;
CPU profiles say nothing about accelerator throughput; local sweeps span too
little of `N` and `D` for a trustworthy fit; contamination of the sweep's own
validation split is not re-checked by the sweep; and every projection is only
as good as its operator inputs.
