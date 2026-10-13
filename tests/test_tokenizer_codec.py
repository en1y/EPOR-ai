"""Byte-exact round trips and control-token safety for the EPOR tokenizer codec."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from tokenizers.pre_tokenizers import ByteLevel

from epor.tokenizer import EporTokenizer, ReservedToken
from epor.tokenizer.artifact import BYTE_TO_UNICODE
from epor.tokenizer.spec import BOS_ID, EOS_ID, FIRST_MERGE_ID, RESERVED_TOKENS

PROPERTY_SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

CODE_SAMPLES = [
    "def f(x):\n\treturn x  \n",
    "if (a) {\r\n    b();\r\n\t  c();   \r\n}\r\n",
    "\t \t mixed\n  \tindentation\t\n\n\n",
    "    trailing spaces    \n\n   ",
    "Ελληνικά 日本語 café naïve — “quotes” …\n",
    "emoji 🙂🚀👩🏽‍💻 flags 🇺🇦 zero​width﻿bom\n",
    "",
    " ",
    "\r",
    "\x00\x01\x7f control bytes\x1b[0m",
    "x = '<|bos|>'  # literal control strings in code\n",
]


def test_byte_level_alphabet_is_the_library_alphabet() -> None:
    assert set(BYTE_TO_UNICODE) == set(ByteLevel.alphabet())
    assert len(set(BYTE_TO_UNICODE)) == 256


@PROPERTY_SETTINGS
@given(data=st.binary(max_size=2_048))
def test_arbitrary_bytes_round_trip_exactly(trained_tokenizer: EporTokenizer, data: bytes) -> None:
    ids = trained_tokenizer.encode(data)
    assert trained_tokenizer.decode_bytes(ids) == data
    assert all(token_id >= 256 for token_id in ids)


@PROPERTY_SETTINGS
@given(text=st.text(max_size=1_024))
def test_arbitrary_text_round_trips_exactly(trained_tokenizer: EporTokenizer, text: str) -> None:
    ids = trained_tokenizer.encode(text)
    assert trained_tokenizer.decode(ids, errors="strict") == text
    assert trained_tokenizer.encode(text.encode("utf-8")) == ids


@pytest.mark.parametrize("sample", CODE_SAMPLES)
def test_code_and_whitespace_round_trip_exactly(
    trained_tokenizer: EporTokenizer, sample: str
) -> None:
    ids = trained_tokenizer.encode(sample)
    assert trained_tokenizer.decode(ids, errors="strict") == sample
    assert trained_tokenizer.decode_bytes(ids) == sample.encode("utf-8")


def test_invalid_utf8_is_preserved_byte_for_byte(trained_tokenizer: EporTokenizer) -> None:
    data = b"valid \xff\xfe invalid \xc3 truncated \xed\xa0\x80 surrogate def f():\n"
    ids = trained_tokenizer.encode(data)
    assert trained_tokenizer.decode_bytes(ids) == data
    assert 256 + 0xFF in ids


def test_learned_merges_compress_the_training_domain(trained_tokenizer: EporTokenizer) -> None:
    text = "def tokenize(value: str) -> list[int]:\n"
    ids = trained_tokenizer.encode(text)
    assert len(ids) < len(text.encode("utf-8"))
    assert any(token_id >= FIRST_MERGE_ID for token_id in ids)


@pytest.mark.parametrize("token", RESERVED_TOKENS, ids=lambda item: item.content)
def test_literal_reserved_strings_never_encode_to_reserved_ids(
    trained_tokenizer: EporTokenizer, token: ReservedToken
) -> None:
    content = token.content
    for text in (content, f"user says {content} here", content * 3):
        ids = trained_tokenizer.encode(text)
        assert all(token_id >= 256 for token_id in ids)
        assert trained_tokenizer.decode(ids) == text


@PROPERTY_SETTINGS
@given(
    parts=st.lists(
        st.one_of(st.sampled_from([item.content for item in RESERVED_TOKENS]), st.text()),
        max_size=12,
    )
)
def test_mixed_control_strings_never_yield_reserved_ids(
    trained_tokenizer: EporTokenizer, parts: list[str]
) -> None:
    text = "".join(parts)
    ids = trained_tokenizer.encode(text)
    assert all(token_id >= 256 for token_id in ids)
    assert trained_tokenizer.decode(ids, errors="strict") == text


def test_reserved_ids_are_inserted_only_explicitly(trained_tokenizer: EporTokenizer) -> None:
    ids = trained_tokenizer.encode("hello", add_bos=True, add_eos=True)
    assert ids[0] == BOS_ID
    assert ids[-1] == EOS_ID
    assert trained_tokenizer.decode(ids) == "hello"
    with pytest.raises(ValueError, match="byte representation"):
        trained_tokenizer.decode_bytes(ids, skip_special_tokens=False)


def test_codec_reports_identity_and_sizes(trained_tokenizer: EporTokenizer) -> None:
    assert trained_tokenizer.sha256 == trained_tokenizer.tokenizer_id
    assert trained_tokenizer.vocab_size == 1_024
    assert FIRST_MERGE_ID < trained_tokenizer.learned_vocab_size <= 1_024
    assert trained_tokenizer.special_tokens["<|eos|>"] == EOS_ID
    with pytest.raises(ValueError, match="outside the learned vocabulary"):
        trained_tokenizer.decode_bytes([trained_tokenizer.learned_vocab_size])
    with pytest.raises(ValueError, match="Unicode scalar"):
        trained_tokenizer.encode("lone \ud800 surrogate")


def test_encode_batch_matches_single_encoding(trained_tokenizer: EporTokenizer) -> None:
    assert trained_tokenizer.encode_batch(CODE_SAMPLES) == [
        trained_tokenizer.encode(sample) for sample in CODE_SAMPLES
    ]
