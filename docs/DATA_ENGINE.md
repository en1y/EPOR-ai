# Provenance-first data engine

Version 0.0.3 implements the offline data contract that precedes tokenizer and
training-dataset work. It does not crawl, call a model API, tokenize data, or
publish a dataset. Every input is a reviewed local file beneath
`EPOR_PROJECT_ROOT`, and every mutation passes the same covenant and authority
checks as other EPOR work.

## Registration before acquisition

Copy [`configs/data/source-registration.example.yaml`](../configs/data/source-registration.example.yaml)
and replace every fixture value with reviewed evidence. A registration records
the owner or steward, canonical origin, acquisition method and time, license or
permission basis, allowed uses, constraints, sensitive-content risk, stable
content family, and removal contact. HTTP(S) origins additionally require both
a `robots` and a `terms` evidence record with a timestamp and SHA-256.

`unknown` is a valid finding, not permission. A source without `train` in
`allowed_uses` may be registered for audit but its documents are quarantined
from builds.

## Immutable layers

The data root defaults to `.epor/data` and may be moved with `EPOR_DATA_ROOT`.
It is created with mode `0700`; files written by the engine are private. The
engine maintains these append-only, content-addressed layers:

1. `registrations/` — one immutable rights record per source ID;
2. `raw/sha256/` — streamed raw bytes addressed by SHA-256;
3. `records/` and `normalized/` — the versioned normalized document contract;
4. `quarantine/` — rights-ineligible, unsafe, executable, empty, or extremely
   low-quality documents;
5. `tombstones/` — source, document, or raw-hash removal directions;
6. `builds/` — deterministic filtered indexes, split JSONL, manifests, and
   dataset cards.

Re-ingesting identical bytes is idempotent. Reusing a source ID with different
registration bytes is rejected; an amendment needs a new source ID. A build
never overwrites a parent.

## Policy sequence

Ingestion incrementally hashes and copies each input with a 256 MiB per-file
ceiling. UTF-8 text is normalized without executing embedded content. The
pipeline records versioned language, domain, and quality classifications;
redacts common PII and credential shapes without retaining matched values in
ordinary logs; reports known benchmark markers; and quarantines executable or
explicit malware/exploit fixtures.

Builds apply all matching tombstones, then globally remove exact duplicates and
near duplicates using four-gram 64-bit SimHash. Only after deduplication do they
assign train/validation/test splits from a salted stable content-family hash.
Repository and benchmark families therefore cannot cross splits merely because
they arrived through different files.

The dataset card records intended and prohibited uses, source and rights
categories, deterministic time range, language/domain mixture, pipeline and
parent hashes, exact/near-duplicate rates, sensitive findings, benchmark-marker
findings, removal behavior, and known gaps.

## CLI and control jobs

Direct commands authorize before work:

```bash
mkdir -p .epor/import
cp configs/data/source-registration.example.yaml .epor/import/source-registration.yaml
uv run --frozen --no-sync epor data ingest \
  .epor/import/source-registration.yaml data/raw/source.txt
uv run --frozen --no-sync epor data build my-dataset-v1
uv run --frozen --no-sync epor data audit
uv run --frozen --no-sync epor data remove document_id <sha256> \
  --reason "Reviewed removal request"
```

The same operations are available as `data_ingest`, `data_build`,
`data_remove`, and `data_audit` control jobs. The API admits them before parsing
their typed specifications; the worker re-resolves authorization at dispatch,
rechecks path containment, supports cooperative cancellation, and exposes only
sanitized JSON/Markdown reports as job artifacts. The console's **Data engine**
page shows registrations, layer counts, removals, builds, and integrity state.

## Evidence and limitations

The offline detectors are deterministic gates, not claims of production-grade
language identification, PII discovery, malware analysis, or contamination
proof. Dataset cards state this explicitly and sampled human review remains
required for both accepted and rejected material. SimHash must be revalidated
before target-scale ingestion. Token shards and the original tokenizer remain
v0.0.4 work; trained weights and dataset publication are outside v0.0.3.
