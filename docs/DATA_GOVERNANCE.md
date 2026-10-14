# Data governance

No corpus is eligible for EPOR training merely because it is publicly reachable.
Registration, acquisition, processing, training, and release are distinct
decisions with separate evidence.

## Admission record

Every source registration must identify the owner or steward, canonical origin,
acquisition method and time, license or permission basis, allowed uses,
geographic or contractual constraints, robots/terms snapshot where applicable,
expected sensitive-content risk, and a removal contact. `unknown` is valid as a
temporary finding, but an unknown right is not permission to train or release.

No autonomous crawl occurs in v0.0.3. Acquisition is limited to contained local
files named by a reviewed registration, with a 256 MiB per-file streaming cap
and immutable request/audit records. A future crawler must add bounded scopes,
rate limits, and separate covenant review before it becomes executable.

## Layer contract

Data moves through immutable, content-addressed layers:

1. **raw** — acquired bytes plus transport and rights metadata;
2. **normalized** — canonical document schema without changing meaning;
3. **filtered** — admitted documents after policy and quality checks;
4. **rejected/quarantined** — retained only where lawful and necessary for audit;
5. **tokenized** — tokenizer-pinned shards preserving document boundaries.

Since v0.0.4 the tokenized layer is implemented: `tokenizers/`, `tokenized/`,
and `evaluations/` beneath the same private data root hold content-addressed
tokenizer artifacts, `uint16` token shards with span indexes, and fertility
reports. A tokenizer records the build and train-split hash it learned from and
refuses a build whose train split has since been tombstoned. Shards and reports
re-apply every current tombstone and record the excluded count and tombstone
IDs; `affected_tokenized_builds` lists the stored shards containing a removed
document so they can be rebuilt and retired. See [`TOKENIZER.md`](TOKENIZER.md).

A derived build records all parent hashes, tool versions, policy/config hashes,
timestamps, counts, and reason-coded removals. Rebuilding creates a new manifest;
it never overwrites a parent.

The implemented v0.0.3 root is private `.epor/data` by default. Registrations,
raw objects, normalized records, quarantines, tombstones, filtered indexes,
split JSONL, manifests, cards, tokenizers, token shards, and fertility reports
are ignored runtime data. Only schemas,
examples, tests, and aggregate documentation belong in Git.

## Required controls

- Exact and global near-duplicate detection runs before split assignment.
- Train/validation/test splits derive deterministically from stable content
  hashes, with repository-family and benchmark-family grouping to prevent
  leakage.
- PII and credentials are detected and removed or quarantined with sampled human
  audit. Secret fixtures include API keys, private keys, tokens, connection
  strings, and high-entropy credentials.
- Malware, exploit corpora, and other unsafe content are isolated behind a
  separate approval and access boundary; general pretraining does not silently
  absorb a quarantine.
- Code ingestion preserves repository topology and dependency order while
  excluding incompatible licenses, generated/vendor trees, secrets, and known
  benchmark solutions.
- Quality, language, and domain classifiers record versions and confidence;
  automated scores are not treated as rights judgments.
- Removal requests create tombstones keyed by stable identifiers. Tombstones
  propagate to all future builds and continued-pretraining lineages.

The v0.0.3 classifiers and detectors are deterministic offline heuristics. They
record versions, confidence, finding kinds, and aggregate counts without
claiming complete language, PII, malware, or benchmark-contamination coverage.
Their accepted and rejected samples still require human audit.

## Evaluation data and contamination reports

Public benchmarks are never downloaded by EPOR. An operator who has reviewed a
benchmark's rights places a local JSONL file in the project and registers it
like any other source; the evaluation harness loads it only when the
registration's `allowed_uses` includes `evaluate`, its media type is
`application/jsonl`, no tombstone removes the source or the file's hash, and
any hash pinned by the suite matches. Training eligibility is a separate
`allowed_uses` entry: registering a benchmark for evaluation does not admit it
to training builds. Evaluation reports record the source ID and the file and
registration hashes, never the local path.

Contamination reports (v0.0.6) measure n-gram overlap between evaluation items
and a training build at the word and token level, and `split-leakage` mode
measures train-split overlap inside a build's validation and test documents.
They hold item IDs, text hashes, counts, and offsets, never item or training
text. A clean scan is a lower bound, not a clearance: paraphrase and
reformatting are not detected, flagged items need human review, and
contamination found in a target run is a mandatory abort condition. See
[`EVALUATION.md`](EVALUATION.md).

## Initial proxy mixture

The starting hypothesis is 55% rights-cleared general/reference English, 25%
permissively licensed code and repository context, 15% math/science/technical
material, and 5% high-quality multilingual material. This is a versioned
experiment, not a fixed recipe. Ablations must report mixture changes, effective
tokens, repeated-token rates, and downstream/safety regressions. v0.0.6
mixture ablations materialize a mixture as a deterministic, non-repeating
subset of one tokenized build's training documents, record unique-token epochs
per subset, and compare per-domain validation loss on the unweighted validation
split (see [`SCALING.md`](SCALING.md)); the choice of mixture stays a human
decision from those measurements.

## Teacher and synthetic data

Every teacher output records provider/model/version, prompt-template hash,
sampling settings, time, applicable terms, cost approval, and the source lineage
of any material placed in the prompt. A filter or judge decision is data and
receives the same lineage treatment.

Synthetic reasoning is admitted only when the task and answer are lawful and
the result can be audited. Compiler/test/SymPy verification is preferred for
code and math. Human review samples both accepted and rejected records. External
model weights are never copied into EPOR.

v0.0.7 implements this as immutable, content-addressed alignment records under
`<data-root>/alignment/records/` (see [`ALIGNMENT.md`](ALIGNMENT.md)). Records
come from four places:

- local EPOR checkpoints, identified by checkpoint, config, and tokenizer hash,
  with the exact sampled tokens;
- operator-registered teacher output files;
- operator-registered training sources;
- project-authored fixtures.

A registered teacher needs two things. First, a data-engine registration that
allows `train`, has media type `application/jsonl`, and no tombstone. Second, a
teacher registration stating its provider, model, version, licence, that the
licence permits training on its outputs, and its cost basis; a paid source
must name its cost-approval record. Registration is CLI-only and immutable,
and EPOR never calls a model API.

AI judge decisions and model-written evaluation items are candidates. A judging
run's candidates reach a dataset only after a verified human has audited a
seeded sample of both its accepted and rejected decisions, and the judge-human
disagreement rate is within a threshold labelled `hypothesis`. A human
rejection excludes its own candidate.

Removal propagates. The data engine's `alignment_record` tombstone, and any
source, document, or raw-hash tombstone in a record's lineage, removes that
record and everything derived from it. Dataset builds exclude removed records
and count them, and training refuses a dataset that holds a record removed
after the dataset was built.

## Privacy and removals

Minimize personal data at acquisition, restrict access by layer, avoid writing
sensitive samples to ordinary logs, and publish only aggregate audit results.
Removal handling must identify affected dataset builds, token shards, runs, and
released derivatives. Whether a trained weight requires withdrawal or a future
mitigation is a documented release/legal decision, never silently ignored.

## Release gate

A dataset card must state intended and prohibited uses, sources and rights
categories, time range, language/domain mixture, pipeline versions, exact and
near-duplicate rates, PII/secret findings, benchmark-overlap findings, removal
process, known gaps, and immutable hashes. Dataset and model licenses are not
inherited from the project's code terms.
