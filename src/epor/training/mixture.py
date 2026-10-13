"""Domain-weighted training subsets of one tokenized build (data-mixture ablations).

A :class:`DomainMixture` names target *token* shares per data-engine domain
label (``DatasetDocument.domain``).  :func:`select_mixture_documents` turns it
into a deterministic subset of one split's documents:

1. Weights are normalized; a domain with weight zero is excluded, and a
   weighted domain absent from the split is an error (never silently empty).
2. Each document's size is its next-token pair count in the token shards.
3. The achievable budget is ``B = min_d T_d / w_d`` over weighted domains,
   where ``T_d`` is the domain's total pairs: the largest subset that meets
   every share *without repeating a document*.  Each domain's target is
   ``w_d * B``.
4. Within a domain, documents are ranked by ``sha256("{seed}:{document_id}")``
   and taken while the domain's running total is below its target (so a
   domain overshoots by less than one document; the scarcest domain is taken
   whole).
5. The subset keeps the split's stable shard order, so the training stream's
   seeded shuffle applies to it exactly as to a full split.

Subsampling rather than repeating keeps unique-token counts honest: two
mixtures trained for the same number of steps differ in composition and in
how often they revisit documents, and both facts are recorded
(:class:`MixtureSelection`), never hidden.  This is an ``EPOR-adaptation``;
whether any mixture is better is decided only by measured validation loss.

The selection is computed from content-addressed inputs alone, so every rank
(and every resume) derives the identical subset.  Validation is never
re-weighted: per-domain validation loss is measured on each domain's full
validation documents.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from epor.data.store import DataStore, canonical_json_bytes, sha256_bytes
from epor.tokenizer.corpus import Split, load_build

if TYPE_CHECKING:
    from .shard_data import ShardCorpus

DomainLabel = Annotated[str, Field(min_length=1, max_length=80)]


class DomainMixture(BaseModel):
    """Target token shares per data-engine domain label, plus the subset seed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    weights: dict[DomainLabel, float]
    seed: int = Field(default=0, ge=0)

    @field_validator("weights")
    @classmethod
    def weights_are_shares(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("a mixture needs at least one domain weight")
        if any(not math.isfinite(weight) or weight < 0 for weight in value.values()):
            raise ValueError("domain weights must be finite and non-negative")
        if not any(weight > 0 for weight in value.values()):
            raise ValueError("at least one domain weight must be positive")
        return dict(sorted(value.items()))

    def normalized(self) -> dict[str, float]:
        total = math.fsum(self.weights.values())
        return {domain: weight / total for domain, weight in self.weights.items() if weight > 0}

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self))


class DomainShare(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    domain: str
    target_share: float
    realized_share: float
    selected_documents: int
    available_documents: int
    selected_pairs: int
    available_pairs: int


class MixtureSelection(BaseModel):
    """The realized subset: document IDs (stable split order) and per-domain shares."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mixture_sha256: str
    tokenized_id: str
    dataset_id: str
    build_id: str
    split: Split
    document_ids: tuple[str, ...]
    selected_pairs: int
    domains: tuple[DomainShare, ...]

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self))

    def summary(self) -> dict[str, object]:
        """Manifest-sized description (no document list)."""

        payload = self.model_dump(mode="json", exclude={"document_ids"})
        payload["selection_sha256"] = self.sha256
        payload["selected_documents"] = len(self.document_ids)
        return payload


def document_domains(store: DataStore | Path, dataset_id: str, build_id: str) -> dict[str, str]:
    """``{document_id: domain}`` for every document of a data-engine build."""

    data_store = store if isinstance(store, DataStore) else DataStore(store)
    build = load_build(data_store, dataset_id, build_id)
    return {document.document_id: document.domain for document in build.documents}


def _rank(seed: int, document_id: str) -> str:
    return hashlib.sha256(f"{seed}:{document_id}".encode()).hexdigest()


def select_from_corpus(
    corpus: ShardCorpus, domains: Mapping[str, str], mixture: DomainMixture
) -> MixtureSelection:
    """Apply ``mixture`` to an opened split (see the module docstring)."""

    shares = mixture.normalized()
    available: dict[str, list[tuple[int, str, int]]] = {}
    for index, (document_id, pairs) in enumerate(
        zip(corpus.document_ids, corpus.pair_counts, strict=True)
    ):
        domain = domains.get(document_id)
        if domain is None:
            raise ValueError(f"document {document_id} is not in the tokenized build's data build")
        available.setdefault(domain, []).append((index, document_id, pairs))
    missing = sorted(set(shares) - set(available))
    if missing:
        raise ValueError(
            f"mixture weights domain(s) absent from the {corpus.split} split: " + ", ".join(missing)
        )
    totals = {domain: sum(item[2] for item in available[domain]) for domain in shares}
    budget = min(totals[domain] / share for domain, share in shares.items())
    chosen: set[int] = set()
    selected_pairs: dict[str, int] = {}
    selected_docs: dict[str, int] = {}
    for domain, share in shares.items():
        target = share * budget
        running = 0
        count = 0
        for index, _document_id, pairs in sorted(
            available[domain], key=lambda item: _rank(mixture.seed, item[1])
        ):
            if running >= target:
                break
            chosen.add(index)
            running += pairs
            count += 1
        selected_pairs[domain] = running
        selected_docs[domain] = count
    total = sum(selected_pairs.values())
    order = sorted(chosen)
    manifest = corpus.manifest
    return MixtureSelection(
        mixture_sha256=mixture.sha256,
        tokenized_id=corpus.tokenized_id,
        dataset_id=manifest.dataset_id,
        build_id=manifest.build_id,
        split=corpus.split,
        document_ids=tuple(corpus.document_ids[index] for index in order),
        selected_pairs=total,
        domains=tuple(
            DomainShare(
                domain=domain,
                target_share=shares.get(domain, 0.0),
                realized_share=(selected_pairs.get(domain, 0) / total) if total else 0.0,
                selected_documents=selected_docs.get(domain, 0),
                available_documents=len(available[domain]),
                selected_pairs=selected_pairs.get(domain, 0),
                available_pairs=sum(item[2] for item in available[domain]),
            )
            for domain in sorted(available)
        ),
    )


def select_mixture_documents(
    store: DataStore | Path,
    tokenized_id: str,
    mixture: DomainMixture,
    *,
    split: Split = "train",
) -> MixtureSelection:
    """Open ``split`` of a tokenized build and apply ``mixture`` to it."""

    from .shard_data import ShardCorpus

    data_store = store if isinstance(store, DataStore) else DataStore(store)
    corpus = ShardCorpus.open(data_store, tokenized_id, split)
    manifest = corpus.manifest
    return select_from_corpus(
        corpus, document_domains(data_store, manifest.dataset_id, manifest.build_id), mixture
    )


__all__ = [
    "DomainMixture",
    "DomainShare",
    "MixtureSelection",
    "document_domains",
    "select_from_corpus",
    "select_mixture_documents",
]
