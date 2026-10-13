# Evaluation harness

v0.0.6 adds a versioned evaluation harness (`epor.evaluation`): suites of
tasks scored against a kernel checkpoint, calibration, a defense-in-depth code
sandbox, contamination and memorization probes, and a model-behaviour safety
suite for the six G10 situations. Every report is content-addressed and every
metric carries its scope.

**Claim boundary.** Every result so far is a `harness-fixture` or
`proxy-scale` measurement. A `harness-fixture` result only shows that the
harness code path works on project-authored items; a `proxy-scale` result
measures a small local checkpoint on named data. Neither is a benchmark score
or a capability claim about any EPOR model, and a safety-suite score on a tiny
model is not evidence of safety.

## Commands and surfaces

```text
epor eval suite CHECKPOINT [--suite fixtures|safety|PATH] [--source ID=PATH]...
                [--training-data TOKENIZED_ID] [--batch-size N] [--max-length N]
                [--allow-code-execution [--allow-unisolated-network]]
epor eval contamination --training-data TOKENIZED_ID [--suite ...] [--source ID=PATH]...
epor eval compare BASELINE_ID CANDIDATE_ID [--resamples N] [--confidence C] [--seed S]
epor eval reports [REPORT_ID]
```

`CHECKPOINT` is a kernel run directory, its `checkpoints/` directory, or one
`step-*` directory. Every operator path must resolve beneath
`EPOR_PROJECT_ROOT`. Output is JSON; progress goes to stderr. Exit codes: `0`
success, `2` input or verification failure (including a tampered report), `3`
refusal by the covenant or by a safety control (an unregistered benchmark
source, or a sandbox that cannot isolate the network without the separate
opt-in), `130` a cooperative stop after SIGINT/SIGTERM, `1` anything else.

| Surface | Action | Notes |
| --- | --- | --- |
| `eval suite`, job `eval_suite` | `eval_suite` | Jobs skip code tasks; they have no field to enable them. |
| `--allow-code-execution` | `eval_code_execution` | CLI only; resolved in addition to `eval_suite`. |
| `eval contamination`, job `eval_contamination` | `eval_contamination` | Runs no model. |
| `eval compare`, `eval reports` | `report_review` | Read-only; verifies every report it reads. |

The console's Evaluation page lists reports from
`GET /api/v1/evaluation/reports`, opens one from
`GET /api/v1/evaluation/reports/{id}` (the verified report without per-item
details such as generated text, plus each task's item count), and compares two
from `GET /api/v1/evaluation/compare`. A report that does not verify is named
as an integrity error (or answered `409`) rather than shown.

## Suites

A suite is versioned YAML (`schema_version: 1`, `id`, `version`, `scope`,
`title`, `seed`, `tasks`). Two ship under `configs/evaluation/`:

- `harness-fixtures.yaml` (`epor-harness-fixtures`) — project-authored
  multiple-choice, exact-match, math, and code items plus the checkpoint's
  validation loss. Not a benchmark.
- `safety-behaviour.yaml` (`epor-safety-behaviour`) — the G10 model-behaviour
  suite described below.

A `harness-fixture` suite may hold only inline project-authored items. A
`proxy-scale` suite may reference operator-registered benchmark files. Control
jobs accept only the two built-in suites or a file under `configs/evaluation/`.

### Task types and scoring choices

| Type | Format | Primary metric | Scoring |
| --- | --- | --- | --- |
| `multiple-choice` | `mc-v1` | `accuracy_norm` | Choice with the highest log-likelihood per UTF-8 byte of the continuation (tokenizer independent); raw summed `accuracy` is reported too. |
| `exact-match` | `exact-match-v1` | `exact_match` | Greedy generation, normalized, must equal one normalized reference. |
| `math` | `math-v1` | `equivalent` | Final answer extracted from greedy output, checked for exact symbolic equivalence without `eval`. |
| `code` | `code-v1` | `pass_at_1` | HumanEval-style prompt + completion + tests in the sandbox; off unless separately enabled. |
| `validation-loss` | — | `main_loss` | The checkpoint's main loss and perplexity on its own build's split. |
| `safety-preference` | `safety-pref-v1` | `preference_norm` | Covenant-consistent continuation must have the higher per-byte log-likelihood. |
| `safety-rubric` | `safety-rubric-v1` | `rubric_pass` | Deterministic phrase rubric on greedy output; a weak proxy. |
| `contamination` | `ngram-contamination-v1` | `flagged_rate` | See below. |
| `memorization` | — | `extraction_gap` | See below. |
| `canary-exposure` | — | `mean_exposure_bits` | See below. |

Per-byte normalization is an `EPOR-adaptation` chosen so scores do not depend
on the tokenizer. Context and continuation are encoded separately; kernel
checkpoints are scored with their build's tokenizer behind `DOC_START`, and
generation is greedy, restricted to decodable IDs, and stops at a stop ID, a
stop string, or the length limit.

The math checker never passes text to `sympify`, `parse_expr`, or anything
else that calls `eval`: a closed recursive-descent grammar builds SymPy
objects directly and bounds length, nesting, literal size, and powers, so
answers such as `__import__('os')` or `9^9^9` are rejected rather than run.

### Calibration

Multiple-choice tasks report expected calibration error over equal-width bins
(default 10) of the largest normalized choice probability (the softmax of the
per-byte scores), the multi-class Brier score in `[0, 2]`, and the reliability
bins themselves (count, mean confidence, accuracy). Ties resolve to the lowest
choice index. The console shows the bins with ECE and Brier.

### Reports and comparison

A report records the suite and its hash, the model identity (config and
checkpoint hashes, step, parameters, tokenizer, scoring settings), data
lineage (project fixtures, registered sources by ID and content hash, tokenized
builds), seeds, harness revision and lock hash, every task result, and the
tasks it skipped with a reason. It carries no timestamps, so identical inputs
give an identical `report_id` — the SHA-256 of the canonical JSON — and it is
stored once at `<data-root>/evaluation-reports/<report_id>.json`. Loading
re-derives the ID and refuses edited bytes.

`eval compare` reports metric deltas for shared tasks and a seeded paired
bootstrap of the mean per-item score difference with a percentile interval.
It warns when suites, suite definitions, tokenizers, scopes, harness
revisions, or evaluation data differ, or when a task is present in only one
report; a warning means the numbers are not directly comparable.

## Registered benchmark sources

EPOR never downloads HumanEval, MBPP, GSM8K, MATH, MMLU, or any other public
benchmark. An operator who has reviewed a benchmark's rights places a local
JSONL file in the project and registers it with the data engine. A suite then
names it by `source_id` and format, optionally pinning its SHA-256 and a
first-N limit, and the command maps the ID to the file with `--source ID=PATH`.
Loading refuses unless the registration exists, its `allowed_uses` includes
`evaluate`, its media type is `application/jsonl`, no tombstone removes the
source or the file's hash, the file is a regular non-symlink file within the
size limit, and any pinned hash matches. The report records the source ID and
the file and registration hashes, never the local path. Supported formats are
`multiple-choice-jsonl`, `exact-match-jsonl`, `gsm8k-jsonl`,
`competition-math-jsonl`, `humaneval-jsonl`, and `mbpp-jsonl`.

## Code execution sandbox

Code tasks execute model-generated Python, so they are off by default
everywhere and are a separate covenant action, `eval_code_execution`, that
only the owner's CLI can request. A delegation scoped to `eval_suite` does not
include it, and no control-job specification can ask for it.

**The sandbox is defense in depth against accidents — an infinite loop, a fork
bomb, a memory hog, a stray write, an unexpected network call — not a security
boundary against a determined attacker.** Run code evaluation only on a machine
and account you are prepared to lose, and prefer a disposable VM or container
for anything beyond the project's own fixtures.

Layers applied to every candidate:

- a fresh `python -I -S -B` process (no user site, no `PYTHON*` environment, no
  script-directory imports, no bytecode) in its own session;
- a private `0700` working directory deleted afterwards, and a separate
  parent-only directory for captured output;
- a stripped environment: `PATH`, `LANG`, and `HOME`/`TMPDIR` pointing at the
  working directory; nothing inherited, including secrets;
- `/dev/null` stdin; stdout and stderr capped by `RLIMIT_FSIZE` and truncated;
- rlimits on CPU seconds, address space, file size, open files, and processes,
  and no core dumps;
- a wall-clock timeout that kills the whole process group, which is also killed
  after a normal exit.

Where the kernel permits, a trusted launcher additionally applies an
unprivileged user and network namespace (the candidate sees only a down
loopback interface, and the process limit counts only that namespace) and
Landlock (writes confined to the working directory; TCP bind/connect denied
with ABI 4+; signals and abstract sockets scoped with ABI 6+). What was
achieved is recorded per task in the report. If the network namespace cannot
be created the run is refused (exit `3`) unless the operator also passes
`--allow-unisolated-network`. On kernels that allow `unshare()` but then deny
the ID mapping (Ubuntu's AppArmor restriction on unprivileged user namespaces),
the launcher stops before the candidate runs and the sandbox retries once
without namespaces, noting the lost layer; that degraded run still needs the
network opt-in.

Known gaps: reads are not confined (a candidate can read any file the user can
and print it into its captured output); a candidate can `setsid` children out
of the process group (they keep the rlimits, namespaces, and Landlock domain);
without a user namespace the process limit counts all of the user's processes;
there is no seccomp filter, cgroup, PID, or mount namespace; kernel
vulnerabilities are out of scope. On the reference machine the sandbox achieved
the user and network namespace and Landlock ABI 10; CI runners may not, and the
tests that assert those layers skip with a reason when the facility is absent.

## Contamination

`epor eval contamination` (or a suite's `contamination` task) measures, for
every evaluation item, the share of its units covered by an n-gram that also
occurs in the training build, at two levels: words of the data-engine split
text (NFKC, case-folded, punctuation dropped) and token IDs of the tokenized
build. The default `n = 13` at both levels is an `EPOR-adaptation` of
published practice. An item is flagged when its train-split coverage reaches
`0.5` — a `hypothesis` threshold to be calibrated with human review — and
`any_match` is reported too; items shorter than `min_ngram` units are reported
as not assessed rather than clean. `split-leakage` mode instead measures how
much of each validation and test document is covered by train-split n-grams.

Limits: only verbatim n-gram reuse after normalization is detected, so this is
a **lower bound** — paraphrases, translations, and reformatted equations are
not found. N-grams are truncated 64-bit BLAKE2b hashes, so rare false matches
are possible. Reports hold item IDs, hashes, counts, and offsets, never item or
training text. A finding is reported for human review; a clean scan is never a
clearance. Contamination found in a target run is a mandatory abort condition
(see [`COMPUTE_POLICY.md`](COMPUTE_POLICY.md)).

## Memorization

The extraction probe samples windows from training-split and held-out
documents of the same build with a seeded reservoir, greedily continues each
prefix, and reports the exact-suffix rate and mean longest-common-prefix
fraction for train and held-out windows. The **gap** (train minus held-out) is
the signal, because held-out boilerplate also extracts. Prefixes are decoded
and re-encoded, which can only lower the measured rate. Canary exposure ranks
a generated canary's log-likelihood among random candidates
(`exposure = log2(N + 1) - log2(rank)`); a model that never saw a canary has
expected exposure near one bit. Nothing in v0.0.6 plants canaries in training
data; that is a reviewed operator action deferred to a later version. These
are measurements, not privacy guarantees.

## G10 model-behaviour safety suite

`safety-behaviour.yaml` holds project-authored paired-preference and rubric
items for the six G10 situations: direct harm, harm through inaction, unsafe
or unauthorized instructions, conflicting directions, shutdown and correction,
and self-preservation pressure. Each item names its category, the covenant
priority it exercises, and the obligation key in `configs/covenant-v1.yaml`;
the test suite lints every item against the covenant and against a content
rule that keeps scenarios abstract and benign. Metrics are reported overall,
per category, and per priority, and every task carries the claim note shown
prominently in the console.

This suite measures **model behaviour** on fixed sentences. It is separate
from, and no substitute for, the system-level covenant resolver in
`src/epor/safety.py`, which orders decisions for control-plane actions and is
tested by `tests/test_safety_covenant.py`. A preference between two fixed
sentences is a narrow, gameable signal at any scale, and a passing score on a
small proxy model proves nothing about safety.

## Evidence

Tests: `test_evaluation_report.py`, `test_evaluation_tasks.py`,
`test_evaluation_calibration.py`, `test_evaluation_mathcheck.py`,
`test_evaluation_sandbox.py`, `test_evaluation_contamination.py`,
`test_evaluation_memorization.py`, `test_evaluation_safety_suite.py`,
`test_evaluation_models.py`, `test_laboratory_cli.py`, and
`test_laboratory_control.py`. A local end-to-end run on a scratch build of
project-owned files trained a 40-step tiny kernel and ran the fixtures and
safety suites, the fixtures suite once with code execution (network namespace, Landlock
ABI 10, and the namespaced process limit all achieved on the reference
machine), a contamination scan, and a comparison; every number from it is a harness-fixture measurement of a model
that has learned almost nothing.
