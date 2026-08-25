# Research dossiers

These files are original paraphrases, not mirrors of third-party papers. The
source of record is [`research/catalog.yaml`](../../research/catalog.yaml); raw
downloads and extracted text are ignored local working material.

## Evidence labels

- **`[disclosed]`** — the cited organization or authors explicitly describe a
  design, process, artifact, or limitation. Disclosure is not independent proof.
- **`[reported-result]`** — a source reports an experiment or measurement. It is
  neither independently reproduced nor an EPOR result.
- **`[EPOR-adaptation]`** — a concrete project decision inspired by evidence but
  owned and tested independently by EPOR.
- **`[hypothesis]`** — an unvalidated proposal or causal interpretation that
  needs a named experiment and baseline.
- **`[unknown]`** — public evidence is insufficient; the dossier intentionally
  refuses to infer a proprietary detail.

Every substantive research assertion in a dossier starts with one of these
labels. A sentence may cite more than one label when, for example, it separates
a disclosed mechanism from an EPOR proposal.

## Integrity states

`integrity: pinned` means the tracked ledger contains a reviewed SHA-256 and a
matching local artifact can be reported as `verified`. `integrity: unpinned`
means the local synchronizer may record the observed digest in ignored metadata
for idempotence, but `epor research verify` reports `unpinned` and exits nonzero.
This prevents a mutable remote page from being mistaken for a reviewed archive.

## Dossiers

- [Anthropic](anthropic.md) — historical disclosure, alignment, data repetition,
  context/retrieval, model cards, model-written evaluations, sparse
  autoencoders, and circuit tracing.
- [DeepSeek](deepseek.md) — LLM, MoE, Coder, Math, V2, Coder-V2, V3, R1, V3.2,
  Engram, and official infrastructure repositories.
- [Gemma and MatFormer](gemma-matformer.md) — Gemma 3, Gemma 3n, MatFormer, and
  the bounded EPOR-γ adaptation.

Review dates and URLs can change without a paper's claims changing. A source
update requires a visible catalog diff, digest review, and dossier review; it is
never silently accepted from the network.
