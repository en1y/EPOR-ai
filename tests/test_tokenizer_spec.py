"""Golden tests: reserved tokenizer IDs must never drift silently."""

from __future__ import annotations

from pathlib import Path

import yaml

from epor.tokenizer.spec import (
    BOS_ID,
    BYTE_OFFSET,
    DEFAULT_VOCAB_SIZE,
    EOS_ID,
    FIRST_MERGE_ID,
    PAD_ID,
    RESERVED_SPEC_SHA256,
    RESERVED_SPEC_VERSION,
    RESERVED_TOKENS,
    SPECIAL_TOKEN_IDS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Changing any of these values is a breaking tokenizer change: it requires a new
# RESERVED_SPEC_VERSION, a retrained tokenizer, and a reviewed rationale.
PINNED_SPEC_SHA256 = "34b8258707db4c81e04fccad5aa1a35d99e0c9ca01b77f98649d3a2ce074b320"
PINNED_NAMED_IDS = {
    "<|pad|>": 0,
    "<|bos|>": 1,
    "<|eos|>": 2,
    "<|system|>": 3,
    "<|user|>": 4,
    "<|assistant|>": 5,
    "<|tool|>": 6,
    "<|end_turn|>": 7,
    "<|tool_call_start|>": 8,
    "<|tool_call_end|>": 9,
    "<|tool_result_start|>": 10,
    "<|tool_result_end|>": 11,
    "<|fim_prefix|>": 12,
    "<|fim_middle|>": 13,
    "<|fim_suffix|>": 14,
    "<|doc_start|>": 15,
    "<|doc_end|>": 16,
    "<|retrieval_start|>": 17,
    "<|retrieval_end|>": 18,
}


def test_reserved_spec_hash_and_version_are_pinned() -> None:
    assert RESERVED_SPEC_VERSION == "epor-reserved-v1"
    assert RESERVED_SPEC_SHA256 == PINNED_SPEC_SHA256


def test_named_reserved_ids_are_pinned() -> None:
    for content, token_id in PINNED_NAMED_IDS.items():
        assert SPECIAL_TOKEN_IDS[content] == token_id
        assert RESERVED_TOKENS[token_id].content == content
    assert (PAD_ID, BOS_ID, EOS_ID) == (0, 1, 2)


def test_reserved_block_layout() -> None:
    assert len(RESERVED_TOKENS) == 256
    assert [item.id for item in RESERVED_TOKENS] == list(range(256))
    assert len(SPECIAL_TOKEN_IDS) == 256
    for item in RESERVED_TOKENS[len(PINNED_NAMED_IDS) :]:
        assert item.content == f"<|reserved_{item.id}|>"
        assert item.category == "reserved"
    assert (BYTE_OFFSET, FIRST_MERGE_ID, DEFAULT_VOCAB_SIZE) == (256, 512, 65_536)


def test_target_model_configs_agree_with_the_reserved_spec() -> None:
    for name in ("epor-alpha", "epor-beta", "epor-gamma"):
        config = yaml.safe_load((PROJECT_ROOT / f"configs/models/{name}.yaml").read_text())
        assert config["vocab_size"] == DEFAULT_VOCAB_SIZE
        assert config["pad_token_id"] == PAD_ID
        assert config["bos_token_id"] == BOS_ID
        assert config["eos_token_id"] == EOS_ID
