"""A dependency-free byte tokenizer for offline fixtures and smoke tests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from typing import ClassVar


class DebugByteTokenizer:
    """Reversible UTF-8/byte tokenizer with four stable special IDs.

    This tokenizer is intentionally not a release tokenizer.  Its only purpose is
    to make model, training, and CI paths runnable without network access.
    """

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    BYTE_OFFSET = 4
    VOCAB_SIZE = 260

    special_tokens: ClassVar[dict[str, int]] = {
        "<pad>": PAD_ID,
        "<bos>": BOS_ID,
        "<eos>": EOS_ID,
        "<unk>": UNK_ID,
    }

    @property
    def vocab_size(self) -> int:
        return self.VOCAB_SIZE

    @property
    def sha256(self) -> str:
        specification = {
            "type": "debug-byte-v1",
            "byte_offset": self.BYTE_OFFSET,
            "special_tokens": self.special_tokens,
            "vocab_size": self.VOCAB_SIZE,
        }
        payload = json.dumps(specification, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def encode(
        self,
        value: str | bytes,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        data = value.encode("utf-8") if isinstance(value, str) else bytes(value)
        ids = [byte + self.BYTE_OFFSET for byte in data]
        if add_bos:
            ids.insert(0, self.BOS_ID)
        if add_eos:
            ids.append(self.EOS_ID)
        return ids

    def decode_bytes(self, token_ids: Iterable[int], *, skip_special_tokens: bool = True) -> bytes:
        output = bytearray()
        special_ids = set(self.special_tokens.values())
        for token_id in token_ids:
            token = int(token_id)
            if token in special_ids:
                if skip_special_tokens:
                    continue
                raise ValueError("special tokens do not have a byte representation")
            if not self.BYTE_OFFSET <= token < self.VOCAB_SIZE:
                raise ValueError(f"token ID {token} is outside the debug vocabulary")
            output.append(token - self.BYTE_OFFSET)
        return bytes(output)

    def decode(
        self,
        token_ids: Sequence[int] | Iterable[int],
        *,
        skip_special_tokens: bool = True,
        errors: str = "replace",
    ) -> str:
        return self.decode_bytes(token_ids, skip_special_tokens=skip_special_tokens).decode(
            "utf-8", errors=errors
        )
