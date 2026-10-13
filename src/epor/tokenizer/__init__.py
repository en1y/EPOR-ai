"""EPOR byte-level BPE tokenizer: reserved-ID spec, trainer, artifacts, and codec.

Consumers (token shards, fertility reports) should depend on:

* ``spec`` constants (``RESERVED_TOKENS``, ``BOS_ID``, ``EOS_ID``, ``DOC_START_ID``,
  ``DOC_END_ID``, ``RESERVED_SPEC_SHA256``) for IDs that never drift;
* ``train_tokenizer`` to produce a content-addressed artifact from a build;
* ``load_tokenizer`` / ``EporTokenizer`` to encode and decode verified artifacts;
* ``load_build`` / ``iter_split_documents`` for hash-verified split reads;
* ``build_token_shards`` / ``verify_tokenized_build`` / ``iter_tokenized_documents``
  for content-addressed, boundary-preserving token shards, and
  ``affected_tokenized_builds`` for removal impact;
* ``evaluate_tokenizer`` for held-out fertility reports.
"""

from .artifact import TokenizerArtifactError, TokenizerManifest, read_artifact
from .codec import EporTokenizer, load_tokenizer
from .corpus import CorpusError, SplitDocument, iter_split_documents, load_build
from .evaluate import FertilityMetrics, FertilityReport, evaluate_tokenizer, load_fertility_report
from .shards import (
    DEFAULT_MAX_SHARD_TOKENS,
    AffectedShards,
    TokenizedBuild,
    TokenizedBuildError,
    TokenizedManifest,
    affected_tokenized_builds,
    build_token_shards,
    iter_tokenized_documents,
    load_tokenized_manifest,
    verify_tokenized_build,
)
from .spec import (
    BOS_ID,
    BYTE_OFFSET,
    DEFAULT_VOCAB_SIZE,
    DOC_END_ID,
    DOC_START_ID,
    EOS_ID,
    FIRST_MERGE_ID,
    PAD_ID,
    RESERVED_SPEC_SHA256,
    RESERVED_SPEC_VERSION,
    RESERVED_TOKENS,
    SPECIAL_TOKEN_IDS,
    ReservedToken,
)
from .train import TrainedTokenizer, train_tokenizer

__all__ = [
    "BOS_ID",
    "BYTE_OFFSET",
    "DEFAULT_MAX_SHARD_TOKENS",
    "DEFAULT_VOCAB_SIZE",
    "DOC_END_ID",
    "DOC_START_ID",
    "EOS_ID",
    "FIRST_MERGE_ID",
    "PAD_ID",
    "RESERVED_SPEC_SHA256",
    "RESERVED_SPEC_VERSION",
    "RESERVED_TOKENS",
    "SPECIAL_TOKEN_IDS",
    "AffectedShards",
    "CorpusError",
    "EporTokenizer",
    "FertilityMetrics",
    "FertilityReport",
    "ReservedToken",
    "SplitDocument",
    "TokenizedBuild",
    "TokenizedBuildError",
    "TokenizedManifest",
    "TokenizerArtifactError",
    "TokenizerManifest",
    "TrainedTokenizer",
    "affected_tokenized_builds",
    "build_token_shards",
    "evaluate_tokenizer",
    "iter_split_documents",
    "iter_tokenized_documents",
    "load_build",
    "load_fertility_report",
    "load_tokenized_manifest",
    "load_tokenizer",
    "read_artifact",
    "train_tokenizer",
    "verify_tokenized_build",
]
