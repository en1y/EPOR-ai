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

A derived build records all parent hashes, tool versions, policy/config hashes,
timestamps, counts, and reason-coded removals. Rebuilding creates a new manifest;
it never overwrites a parent.

The implemented v0.0.3 root is private `.epor/data` by default. Registrations,
raw objects, normalized records, quarantines, tombstones, filtered indexes,
split JSONL, manifests, and cards are ignored runtime data. Only schemas,
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

## Initial proxy mixture

The starting hypothesis is 55% rights-cleared general/reference English, 25%
permissively licensed code and repository context, 15% math/science/technical
material, and 5% high-quality multilingual material. This is a versioned
experiment, not a fixed recipe. Ablations must report mixture changes, effective
tokens, repeated-token rates, and downstream/safety regressions.

## Teacher and synthetic data

Every teacher output records provider/model/version, prompt-template hash,
sampling settings, time, applicable terms, cost approval, and the source lineage
of any material placed in the prompt. A filter or judge decision is data and
receives the same lineage treatment.

Synthetic reasoning is admitted only when the task and answer are lawful and
the result can be audited. Compiler/test/SymPy verification is preferred for
code and math. Human review samples both accepted and rejected records. External
model weights are never copied into EPOR.

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
