"""Frozen reserved-ID specification for the EPOR byte-level BPE tokenizer.

The vocabulary layout is fixed and versioned:

* IDs ``0..255`` are reserved control tokens (``RESERVED_TOKENS``). They are never
  produced by encoding ordinary text; a caller inserts them by ID.
* IDs ``256..511`` are the 256 byte tokens, in byte order: byte ``b`` is ID
  ``256 + b``. Every byte sequence is therefore always encodable (byte fallback).
* IDs ``512..vocab_size-1`` are learned merges, in merge order.

Named reserved tokens:

=====  ========================  ==========  ==============================================
ID     Content                   Category    Purpose
=====  ========================  ==========  ==============================================
0      ``<|pad|>``               control     Padding; matches model configs' pad_token_id.
1      ``<|bos|>``               control     Beginning of sequence; bos_token_id.
2      ``<|eos|>``               control     End of sequence; eos_token_id.
3      ``<|system|>``            chat        Opens a system-role message.
4      ``<|user|>``              chat        Opens a user-role message.
5      ``<|assistant|>``         chat        Opens an assistant-role message.
6      ``<|tool|>``              chat        Opens a tool-role message.
7      ``<|end_turn|>``          chat        Closes any role message.
8      ``<|tool_call_start|>``   tool        Opens a structured tool call.
9      ``<|tool_call_end|>``     tool        Closes a structured tool call.
10     ``<|tool_result_start|>`` tool        Opens a tool result.
11     ``<|tool_result_end|>``   tool        Closes a tool result.
12     ``<|fim_prefix|>``        fim         Fill-in-the-middle prefix marker.
13     ``<|fim_middle|>``        fim         Fill-in-the-middle middle (target) marker.
14     ``<|fim_suffix|>``        fim         Fill-in-the-middle suffix marker.
15     ``<|doc_start|>``         document    Document boundary: start.
16     ``<|doc_end|>``           document    Document boundary: end.
17     ``<|retrieval_start|>``   retrieval   Opens retrieved context.
18     ``<|retrieval_end|>``     retrieval   Closes retrieved context.
19-255 ``<|reserved_N|>``        reserved    Future extensions; N is the token's own ID.
=====  ========================  ==========  ==============================================

Changing any entry changes ``RESERVED_SPEC_SHA256`` and must ship as a new spec
version: tokenizer artifacts pin the hash, and IDs must never drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from epor.data.store import canonical_json_bytes, sha256_bytes

Category = Literal["control", "chat", "tool", "fim", "document", "retrieval", "reserved"]

RESERVED_SPEC_VERSION: Final = "epor-reserved-v1"
RESERVED_COUNT: Final = 256
BYTE_OFFSET: Final = RESERVED_COUNT
BYTE_COUNT: Final = 256
FIRST_MERGE_ID: Final = BYTE_OFFSET + BYTE_COUNT
DEFAULT_VOCAB_SIZE: Final = 65_536
# Fixture tokenizers must still be able to learn at least a few merges.
MIN_VOCAB_SIZE: Final = FIRST_MERGE_ID + 8


@dataclass(frozen=True, slots=True)
class ReservedToken:
    id: int
    content: str
    category: Category


_NAMED: Final[tuple[tuple[str, Category], ...]] = (
    ("<|pad|>", "control"),
    ("<|bos|>", "control"),
    ("<|eos|>", "control"),
    ("<|system|>", "chat"),
    ("<|user|>", "chat"),
    ("<|assistant|>", "chat"),
    ("<|tool|>", "chat"),
    ("<|end_turn|>", "chat"),
    ("<|tool_call_start|>", "tool"),
    ("<|tool_call_end|>", "tool"),
    ("<|tool_result_start|>", "tool"),
    ("<|tool_result_end|>", "tool"),
    ("<|fim_prefix|>", "fim"),
    ("<|fim_middle|>", "fim"),
    ("<|fim_suffix|>", "fim"),
    ("<|doc_start|>", "document"),
    ("<|doc_end|>", "document"),
    ("<|retrieval_start|>", "retrieval"),
    ("<|retrieval_end|>", "retrieval"),
)

RESERVED_TOKENS: Final[tuple[ReservedToken, ...]] = tuple(
    ReservedToken(id=index, content=content, category=category)
    for index, (content, category) in enumerate(_NAMED)
) + tuple(
    ReservedToken(id=index, content=f"<|reserved_{index}|>", category="reserved")
    for index in range(len(_NAMED), RESERVED_COUNT)
)

SPECIAL_TOKEN_IDS: Final[dict[str, int]] = {item.content: item.id for item in RESERVED_TOKENS}

PAD_ID: Final = SPECIAL_TOKEN_IDS["<|pad|>"]
BOS_ID: Final = SPECIAL_TOKEN_IDS["<|bos|>"]
EOS_ID: Final = SPECIAL_TOKEN_IDS["<|eos|>"]
DOC_START_ID: Final = SPECIAL_TOKEN_IDS["<|doc_start|>"]
DOC_END_ID: Final = SPECIAL_TOKEN_IDS["<|doc_end|>"]


def reserved_spec() -> dict[str, object]:
    """The canonical, hashable description of the fixed vocabulary layout."""

    return {
        "version": RESERVED_SPEC_VERSION,
        "reserved_count": RESERVED_COUNT,
        "byte_offset": BYTE_OFFSET,
        "byte_count": BYTE_COUNT,
        "first_merge_id": FIRST_MERGE_ID,
        "tokens": [
            {"id": item.id, "content": item.content, "category": item.category}
            for item in RESERVED_TOKENS
        ],
    }


RESERVED_SPEC_SHA256: Final = sha256_bytes(canonical_json_bytes(reserved_spec()))


def _validate() -> None:
    if len(RESERVED_TOKENS) != RESERVED_COUNT:
        raise AssertionError("reserved table must contain exactly 256 entries")
    if [item.id for item in RESERVED_TOKENS] != list(range(RESERVED_COUNT)):
        raise AssertionError("reserved IDs must be contiguous from zero")
    if len(SPECIAL_TOKEN_IDS) != RESERVED_COUNT:
        raise AssertionError("reserved token contents must be unique")


_validate()
