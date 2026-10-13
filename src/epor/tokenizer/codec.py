"""Verified, byte-exact encoding and decoding with a trained EPOR tokenizer."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final

from tokenizers import Tokenizer

from epor.data.store import DataStore

from .artifact import (
    UNICODE_TO_BYTE,
    TokenizerArtifactError,
    TokenizerManifest,
    read_artifact,
)
from .spec import BOS_ID, BYTE_OFFSET, EOS_ID, PAD_ID, RESERVED_COUNT, SPECIAL_TOKEN_IDS

# ``bytes.decode("utf-8", "surrogateescape")`` maps each undecodable byte to one
# lone surrogate in this range; valid UTF-8 never produces one.
_ESCAPED_BYTES: Final = re.compile("[\udc80-\udcff]+")


class EporTokenizer:
    """A loaded, hash-verified byte-level BPE tokenizer.

    ``encode`` never yields a reserved ID (0-255) from input content, even when
    the input literally contains a string such as ``<|bos|>``; reserved IDs are
    only added explicitly (``add_bos``/``add_eos``) or by a caller that inserts
    them by ID. Arbitrary bytes, including invalid UTF-8, round-trip exactly
    through ``encode`` and ``decode_bytes``.
    """

    def __init__(self, tokenizer_id: str, manifest: TokenizerManifest, tokenizer_json: bytes):
        self._id = tokenizer_id
        self._manifest = manifest
        self._tokenizer = Tokenizer.from_str(tokenizer_json.decode("utf-8"))
        # Not serialized by the library: matching special-token text in the input
        # would let ordinary content forge control tokens.
        self._tokenizer.encode_special_tokens = True
        size = manifest.learned_vocab_size
        self._token_bytes: list[bytes | None] = [None] * size
        for token_id in range(RESERVED_COUNT, size):
            token = self._tokenizer.id_to_token(token_id)
            if token is None:
                raise TokenizerArtifactError(f"tokenizer has no token for ID {token_id}")
            self._token_bytes[token_id] = bytes(UNICODE_TO_BYTE[char] for char in token)

    @classmethod
    def from_directory(cls, directory: Path) -> EporTokenizer:
        return cls(*read_artifact(directory))

    @property
    def tokenizer_id(self) -> str:
        """Content address of the artifact (SHA-256 of its canonical manifest)."""

        return self._id

    @property
    def sha256(self) -> str:
        return self._id

    @property
    def manifest(self) -> TokenizerManifest:
        return self._manifest

    @property
    def vocab_size(self) -> int:
        """Declared vocabulary size, i.e. the model embedding-table size."""

        return self._manifest.vocab_size

    @property
    def learned_vocab_size(self) -> int:
        """IDs actually defined by the artifact; IDs at or above this are unused."""

        return self._manifest.learned_vocab_size

    @property
    def special_tokens(self) -> dict[str, int]:
        return dict(SPECIAL_TOKEN_IDS)

    pad_id: Final = PAD_ID
    bos_id: Final = BOS_ID
    eos_id: Final = EOS_ID

    def _encode_text(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def _content_ids(self, value: str | bytes) -> list[int]:
        if isinstance(value, str):
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError(
                    "text must consist of Unicode scalar values; pass bytes for raw data"
                ) from exc
            ids = self._encode_text(value)
        else:
            escaped = bytes(value).decode("utf-8", "surrogateescape")
            ids = []
            position = 0
            for match in _ESCAPED_BYTES.finditer(escaped):
                if match.start() > position:
                    ids.extend(self._encode_text(escaped[position : match.start()]))
                ids.extend(
                    BYTE_OFFSET + byte for byte in match.group().encode("utf-8", "surrogateescape")
                )
                position = match.end()
            if position < len(escaped):
                ids.extend(self._encode_text(escaped[position:]))
        if any(token_id < RESERVED_COUNT for token_id in ids):
            # Fail closed: content must never become a control token.
            raise TokenizerArtifactError("encoding produced a reserved ID from content")
        return ids

    def encode(
        self,
        value: str | bytes,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        ids = self._content_ids(value)
        if add_bos:
            ids.insert(0, BOS_ID)
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        """Encode many texts; results are identical to calling ``encode`` per text."""

        return [self._content_ids(text) for text in texts]

    def decode_bytes(self, token_ids: Iterable[int], *, skip_special_tokens: bool = True) -> bytes:
        output = bytearray()
        size = len(self._token_bytes)
        for raw in token_ids:
            token_id = int(raw)
            if not 0 <= token_id < size:
                raise ValueError(f"token ID {token_id} is outside the learned vocabulary")
            piece = self._token_bytes[token_id]
            if piece is None:
                if skip_special_tokens:
                    continue
                raise ValueError("reserved tokens do not have a byte representation")
            output += piece
        return bytes(output)

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
        errors: str = "replace",
    ) -> str:
        return self.decode_bytes(token_ids, skip_special_tokens=skip_special_tokens).decode(
            "utf-8", errors=errors
        )


def load_tokenizer(store: DataStore | Path, tokenizer_id: str) -> EporTokenizer:
    """Load a stored tokenizer by ID from a data store (or its root path)."""

    data_store = store if isinstance(store, DataStore) else DataStore(store)
    directory = data_store.tokenizer_path(tokenizer_id)
    if not directory.is_dir():
        raise TokenizerArtifactError(f"tokenizer artifact not found: {tokenizer_id}")
    return EporTokenizer.from_directory(directory)
