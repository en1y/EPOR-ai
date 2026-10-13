"""EPOR byte-level BPE tokenizer: reserved-ID spec, trainer, artifacts, and codec.

Consumers (token shards, fertility reports) should depend on:

* ``spec`` constants (``RESERVED_TOKENS``, ``BOS_ID``, ``EOS_ID``, ``DOC_START_ID``,
  ``DOC_END_ID``, ``RESERVED_SPEC_SHA256``) for IDs that never drift;
* ``train_tokenizer`` to produce a content-addressed artifact from a build;
* ``load_tokenizer`` / ``EporTokenizer`` to encode and decode verified artifacts;
* ``load_build`` / ``iter_split_documents`` for hash-verified split reads.
"""

from .artifact import TokenizerArtifactError, TokenizerManifest, read_artifact
from .codec import EporTokenizer, load_tokenizer
from .corpus import CorpusError, SplitDocument, iter_split_documents, load_build
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
    "CorpusError",
    "EporTokenizer",
    "ReservedToken",
    "SplitDocument",
    "TokenizerArtifactError",
    "TokenizerManifest",
    "TrainedTokenizer",
    "iter_split_documents",
    "load_build",
    "load_tokenizer",
    "read_artifact",
    "train_tokenizer",
]
