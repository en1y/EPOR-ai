# Knowledge transfer and alignment (v0.0.7)

This document describes the alignment pipeline in `epor.alignment`: how
teacher data is recorded, how supervised fine-tuning, distillation, and the
DPO baseline train, how model feedback is kept a candidate until humans audit
it, and how reward overoptimization and model-level G10 behaviour are
measured.

**Claim boundary.** Everything here runs at proxy scale on a CPU. A tiny model
fine-tuned on a few dozen project-authored sentences, preference-optimized on
a dozen pairs, and compared on the G10 items shows that the pipeline runs and
that its controls hold. It is **not** evidence that any EPOR model is aligned,
helpful, honest, or safe, and no report produced here may be presented as such.

## Commands and actions

| Command | Covenant action | Job type | Notes |
| --- | --- | --- | --- |
| `epor align fixtures` | `align_register` | — | Seeds the project-authored fixtures as records. |
| `epor align teacher register <yaml>` | `align_register` | — | Owner CLI only. |
| `epor align teacher import <teacher> <jsonl>` | `align_register` | — | Owner CLI only. |
| `epor align source import <source> <jsonl>` | `align_register` | — | Owner CLI only. |
| `epor align teacher sample <ckpt>` | `align_generate` | `align_generate` | EPOR checkpoints only. |
| `epor align critique <ckpt>` | `align_generate` | `align_generate` | Cites obligations by key. |
| `epor align eval-items <ckpt>` | `align_generate` | `align_generate` | Candidates for human review. |
| `epor align judge <ckpt>` | `align_judge` | `align_judge` | Decisions are data. |
| `epor align dataset sft\|preference` | `align_dataset` | `align_dataset` | Applies the audit admission rule. |
| `epor train sft` | `align_sft` | `align_sft` | Runs inside the training kernel. |
| `epor train preference` | `align_preference` | `align_preference` | DPO baseline. |
| `epor align audit sample` | `align_audit` | — | Queues items for humans. |
| `epor align audit verdict` | `align_audit` | — | Also `POST /api/v1/alignment/audits/{id}/verdict`. |
| `epor align audit calibration` | `align_evaluate` | `align_evaluate` | Stores a calibration report. |
| `epor align monitor` | `align_evaluate` | `align_evaluate` | Overoptimization report. |
| `epor align g10` | `align_evaluate` | `align_evaluate` | Before/after comparison. |
| `epor align audit list`, `epor align reports` | `report_review` | — | Read-only. |

`epor train grpo` remains a gated placeholder: scope-limited GRPO on
verifiable rewards is a v0.5.0 item.

Each action is its own delegated scope. Pretraining authority does not cover
fine-tuning, SFT does not cover DPO, and neither training nor judging covers
auditing, which grants no training or dataset authority in turn
(`tests/test_safety_covenant.py`). Registering teacher material and recording
verdicts are deliberately not job types.

Exit codes: 0 success; 1 failure, a G10 comparison that fails closed, an
overoptimization report at `abort`, or a DPO run stopped by its monitor; 2
input or verification failure; 3 refusal by the covenant or a rights control;
4 a G10 comparison escalated to human review; 130 stopped by SIGINT/SIGTERM.

## The chat format

`epor-chat-v1` encodes a conversation on the frozen reserved IDs
(`src/epor/tokenizer/spec.py`):

```text
<|bos|> ROLE content <|end_turn|> ... ROLE content <|end_turn|> <|eos|>
```

with `ROLE` one of `<|system|>` (3), `<|user|>` (4), `<|assistant|>` (5), and
`<|tool|>` (6), and `<|end_turn|>` 7. Role and turn tokens are inserted by ID;
the codec refuses to produce a reserved ID from text, so message content
cannot forge a turn or a role. Only assistant content, the `<|end_turn|>` that
closes an assistant turn, and a final `<|eos|>` after an assistant turn are
training targets. The `<|assistant|>` token is supplied by the prompt at
generation time and is never learned. Generation is restricted to learned
byte/merge IDs plus the stop IDs, so a model cannot emit a role token either.
The format's description, pinned to the reserved-spec hash, is hashed as
`CHAT_FORMAT_SHA256` and recorded in every dataset manifest.

## Teacher-data provenance

Every example, judge decision, preference pair, and model-written evaluation
item is an immutable record under `<data-root>/alignment/records/<kind>/`,
addressed by the SHA-256 of its canonical JSON and re-verified on every read.
A record carries:

- **generator** — the project (authored fixtures), a local EPOR checkpoint (its
  `checkpoint.json` hash, config hash, and tokenizer ID), an operator-registered
  teacher output file (provider, model, version), an operator-registered
  training source, or a human;
- **rights** — licence, whether training on the outputs is permitted and on
  what basis, the data-engine registration hash, and the cost basis. A paid
  source must name its cost-approval record; EPOR never calls a model API;
- **prompt template** ID and hash, **sampling** settings (decoding,
  temperature, top-k, reply limit, seed), **time**, the **exact output**, and
  for EPOR teachers the exact output token IDs;
- **lineage** — every fixture, source, document, raw file, checkpoint,
  covenant, or parent record whose material entered the prompt or decision.

Judge decisions get the same treatment as any example. Teachers come in two
kinds, and neither contributes weights:

- **EPOR checkpoints** — sampled locally with `epor align teacher sample`.
  These are the only teachers whose logits may be distilled.
- **Registered output files** — an operator who holds rights to another
  model's outputs registers the local JSONL with the data engine (its
  registration must allow `train` and use `application/jsonl`) and then as a
  teacher source (`TeacherSource`: provider, model, version, licence,
  `training_on_outputs_permitted: true` with its basis, and cost). A teacher
  whose licence does not permit training on outputs cannot be registered.
  Registrations are immutable. Imported outputs serve response distillation
  only.

Operator-curated demonstrations and preference pairs enter the same way with
`epor align source import` (`epor-chat-jsonl` or `epor-preference-jsonl`).
Project fixtures live in `configs/alignment/fixtures.yaml`; a test fails if
any of their text shares a six-word span with the G10 evaluation items, so
training on them cannot contaminate that suite.

### Removal

Data-engine tombstones gain an `alignment_record` target
(`epor data remove alignment_record <id>`). A record is removed when a
tombstone names it or anything in its lineage, transitively: removing a source
removes every output imported from it, the judge decisions about those
outputs, and the preference pairs built from them. Dataset builds exclude
removed records and count them, and training refuses a dataset that holds a
record removed after the dataset was built.

## Datasets

`epor align dataset sft|preference` builds a content-addressed dataset under
`<data-root>/alignment/datasets/<id>/` from named records or record groups
(SFT: `fixtures`, `teacher-outputs`, `revisions`, `registered`; preference:
`fixtures`, `ai-judge`, `registered`, `human`). Records split train/validation
by a salted hash of their ID. Admission rules:

- removed records are excluded and counted;
- a record without training-on-outputs rights is refused, and a registered
  source must still allow training;
- critiques are intermediate text and are never training targets;
- a model-authored candidate (a revision, or an `ai-judge` preference) is
  admitted only through the audit admission rule below, and otherwise excluded
  and counted.

The manifest pins the chat format, reserved-ID spec, tokenizer, every split
file's hash, every record ID, source counts, the admission evidence, and the
exclusion counts. It has no timestamp, so identical inputs give an identical ID.

## Supervised fine-tuning and distillation

SFT runs inside the v0.0.5 kernel (`pretrain_packed`) rather than a parallel
trainer. Three optional `PackedTrainingSettings` fields carry it, each left out
of the trajectory hash when unset, so every earlier checkpoint keeps its hash:

- `chat_dataset_id` trains and validates on a verified chat dataset. Packing
  counts only unmasked targets, so loss normalization covers assistant tokens
  only. A micro-batch whose targets are all masked contributes an exact zero
  that still touches every parameter, so DDP and FSDP2 reduce identically.
- `init_checkpoint` starts step 1 from a pinned kernel checkpoint's weights,
  with the same config and tokenizer, and a fresh optimizer.
- `distillation` adds `weight * T^2 * KL(teacher_T || student_T)` on every
  trained target, against the mean of up to four frozen EPOR teachers. It is
  refused unless every teacher shares the student's tokenizer and vocabulary.

**[disclosed]** Gemma 3 and MatFormer describe distillation during training.
**[EPOR-adaptation]** *Response* (sequence-level) distillation is SFT on
teacher outputs from any registered teacher. Each example keeps its own
teacher record, so a dataset may mix several teachers. *Logit* distillation is
the `distillation` setting above. No external weights ever load into or seed
an EPOR checkpoint. A fresh student (`--recipe ... --vocab-from-build`) takes
the build's vocabulary through the same explicit opt-in as `epor train
kernel`.

Evidence (`tests/test_alignment_sft.py`, `tests/test_alignment_transfer.py`):
assistant-only targets; SFT starts from the parent's weights; exact resume;
exact agreement of single-process, DDP, and FSDP2 SFT; a pinned parent that
does not match is refused; logit distillation resumes exactly and is refused
across tokenizers; and a removal after a build refuses training until the
dataset is rebuilt.

## The DPO baseline

**[disclosed]** DPO fits a policy to preference pairs against a frozen
reference without a separate reward model:

```text
loss = -log sigmoid(beta * ((log pi(y+|x) - log ref(y+|x)) - (log pi(y-|x) - log ref(y-|x))))
```

with log-probabilities summed over the reply and its `<|end_turn|>`.
**[EPOR-adaptation]** It is the v0.0.7 baseline against SFT. The policy starts
from a pinned checkpoint, and the reference is a pinned checkpoint that is
read and never written (by default the same SFT checkpoint). Training is one
process with the kernel's AdamW groups, schedule, clipping, and checkpoint
format. Pairs are drawn from a per-epoch permutation seeded by
`(seed, epoch)`, so resume is exact. Each step records the loss, the implicit
reward margin and accuracy, chosen and rejected rewards, and
`policy_logratio`: the mean per-token `log pi - log ref` on chosen replies.
That is a sample estimate of KL to the reference on the data. Crossing the
alert threshold is recorded; crossing the abort threshold checkpoints and
stops the run as `aborted`. Kernel resume refuses DPO checkpoints, and
cancellation checkpoints the last completed step.

## Constitutional critique and revision

**[disclosed]** Constitutional AI has a model critique its response against a
written principle, then revise it. **[EPOR-adaptation]** The constitution is
`configs/covenant-v1.yaml` itself. `epor.alignment.critique` names
obligations by key, reads their text from the covenant file at run time, and
records the covenant hash and the cited keys on every output; it never restates a
principle. `tests/test_alignment_critique.py` fails if any alignment module or
configuration shares a six-word span with a covenant law or obligation, and
checks that the text used is the covenant file's own. Critiques and revisions
are candidates only. A revision record keeps the original prompt, so an
admitted revision is a demonstration for it.

**[hypothesis]** Model feedback may reduce human exposure to disturbing
material without loss of quality. It must beat the SFT and human-preference
controls before it becomes a default, and nothing in v0.0.7 measures that.

## AI judging, model-written evaluations, and human audits

**[disclosed]** Model feedback can label preferences, and model-written
evaluations generate behavioural test items. **[reported-result]** Such items
inherit the generating model's biases. **[EPOR-adaptation]** Both are
candidate generators only.

The judge scores each reply by its mean per-token log-likelihood under a fixed
prompt that names a covenant obligation by key. A decision is accepted when
the score difference reaches `--min-margin`; for revisions, only when the
revision wins. Every decision is a `JudgementRecord`, and an accepted pairwise
decision also writes an `ai-judge` preference pair. Model-written G10 items
cycle through the six situations and cite an obligation per situation; they
stay candidates until a human accepts them, and the held-out G10 suite stays
human-authored.

### The audit queue

The queue is control-plane truth: an append-only, hash-chained log
(`<safety-root>/alignment-audit.jsonl`, the same primitive as the identity
log). `epor align audit sample <run>` queues a seeded sample of up to N
accepted **and** N rejected decisions, with the prompt, replies, judge pick,
and the question to answer. Items start pending. **Nothing is ever
auto-marked audited.** The only writer of a verdict
(`AuditLedger.record_verdict`) requires an authenticated actor, and the
CLI and API gate it through `align_audit`. No job handler, judge, sampler,
or report calls it. A later verdict supersedes an earlier one without
deleting it. G10 failures also land in the queue as `g10-item` entries for
human review.

### Calibration and admission

`epor align audit calibration <run>` reports the judge–human agreement matrix,
the **disagreement rate** overall and per stratum, and reliability bins of
judge confidence against human agreement. An item agrees when the human
accepts a decision the judge accepted, or rejects one it rejected. A judging
run's candidates may train only when:

- every sampled item has a verdict;
- at least `min_per_stratum` accepted **and** rejected decisions were
  audited;
- the disagreement rate is at most `max_disagreement`.

Within an admitted run, a candidate still needs the judge's acceptance, and a
human rejection of its own decision excludes it. A run whose judge accepted
everything has no rejected stratum to audit, so it can never be admitted.

## Reward-overoptimization monitors

`epor align monitor <policy> --reference <ref> --dataset <preference-id>`
measures three signals and stores a content-addressed report under
`<data-root>/alignment/reports/overoptimization/`:

- `kl_per_token` — mean per-token `log pi - log ref` on the policy's own
  greedy replies to held-out prompts (a sample estimate of KL(pi || ref));
- `proxy_minus_audited` — proxy reward accuracy (the DPO implicit reward
  prefers the chosen reply on held-out pairs) minus the policy's agreement with
  human-endorsed judge decisions. Only `accept` verdicts endorse an ordering;
  a rejection says the judge's pick is not better and may describe two
  equally poor replies. The monitor is `unknown` until a human has endorsed a
  pair;
- `length_ratio` — mean policy reply length over the reference's on the same
  prompts, checked in both directions.

Each signal is classified `ok`, `alert`, or `abort` against
`configs/alignment/monitors.yaml`, and the report takes the worst status.
**Every threshold there is a `[hypothesis]`** and must be revisited against
EPOR's own SFT and human-preference controls:

| Monitor | Alert | Abort |
| --- | --- | --- |
| `kl_per_token` | 0.5 | 2.0 |
| `proxy_minus_audited` | 0.2 | 0.4 |
| `length_ratio` (either direction) | 1.5 | 2.5 |
| DPO per-step `policy_logratio` | 0.5 | 2.0 |
| judge admission: maximum disagreement | 0.25 | — |

An `abort` report blocks the policy from being treated as a promotion
candidate until a human reviews it.

## G10 at model level

v0.0.2 tests G10 on the resolver; v0.0.6 added project-authored
model-behaviour items for the same six situations. `epor align g10 --before
<ckpt> --after <ckpt>` runs those items, unchanged, through the chat format on
the checkpoint before an alignment step and the one after:

- preference items pass when the covenant-consistent reply has the higher
  log-likelihood per UTF-8 byte;
- rubric items take a greedy reply and apply their phrase rubric, a weak proxy.

The gate fails closed or escalates and never reports success. `blocked` means
some situation's pass rate fell by more than the tolerance (exit 1).
`escalated` means nothing regressed but items still fail; each failure is
queued for a human (exit 4). `no-regression-observed` says only that two
fixed sets of sentences compared without regression.

## Console and API

The console's **Alignment** page shows:

- provenance record counts;
- the audit queue, with the material a reviewer needs rendered as inert text
  and Accept/Reject controls for an identity holding `align_audit`;
- judge calibration and admission per judging run;
- the overoptimization monitors;
- the latest G10 comparison.

The training page lists SFT, distillation, and DPO runs beside pretraining,
with their objective.

Read-only endpoints are `GET /api/v1/alignment`,
`/alignment/audits[?state=&run=]`, `/alignment/audits/{id}`, and
`/alignment/reports/{kind}/{id}` (content addresses verified, tampering is a
409). `POST /alignment/audits/{id}/verdict` requires a verified identity
holding `align_audit` and creates no job. Job results carry IDs and small
summaries whose keys avoid the credential pattern, so v0.0.7 adds no
redaction allowlist entry. For example, the overoptimization job result
reports its KL monitor under `kl`.

## End-to-end run (local, proxy scale)

A complete CLI run on a scratch data root, using project documentation as the
corpus (11 documents, a 2,048-ID tokenizer, 35,022 tokens), exercised:

1. **Base model.** Pretraining `epor-tiny` for 400 steps (validation
   perplexity 110.6).
2. **SFT.** 150 steps on the 16 fixture demonstrations. Training loss fell to
   0.23 while held-out loss stayed 5.90: the model memorises 12 examples.
3. **Teachers.** Sampling from two EPOR teachers (base and SFT).
4. **Distillation.** A fresh student by response distillation over both
   teachers plus the fixtures, and logit distillation from the SFT checkpoint
   (final per-token KL 0.58).
5. **Registered teacher file.** Registering and importing one, with no
   external model involved.
6. **Critique and judging.** Critique/revision against two obligations, and
   three judging runs.
7. **Audit.** Six owner verdicts, five through the CLI and one through the
   API; an anonymous API verdict was refused with 401.
8. **Calibration and gate.** Calibration found disagreement 0.5, so the gate
   excluded all 12 AI-judged pairs.
9. **DPO.** 60 steps on the 12 project pairs: 40 steps alerted, none aborted.
10. **Monitor.** The overoptimization report was `unknown`: KL 0.012 and
    length ratio 1.16 were `ok`, and no human-endorsed pair existed.
11. **G10.** Comparisons for base→SFT and SFT→DPO. **Both were blocked** by a
    situation-level regression, and their failures were queued for review.

The judge also showed a strong self-preference: it scored its own samples
10–11.6 nats per token above the base model's degenerate ones, which is why
the audit exists. None of these numbers describes a capable model.

## Known limits

- Every model here is a proxy-scale CPU artifact, and every threshold is a
  hypothesis.
- The judge's per-token likelihood is a weak, gameable signal, and judge
  confidence saturates on tiny models. Calibration bins are therefore
  uninformative until judges are stronger.
- The phrase rubric misses negation and paraphrase.
- DPO runs in one process; SFT and distillation have local DDP/FSDP2 through
  the CLI only.
- The audit queue replays its log on every read. That is adequate for a
  single-user machine; it would need indexing at scale.
- Logit distillation needs teacher and student to share a tokenizer. Teachers
  under other vocabularies serve response distillation only.
