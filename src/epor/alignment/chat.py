"""The ``epor-chat-v1`` conversation format on the frozen reserved role tokens.

A conversation of messages ``m1 .. mk`` encodes as::

    <|bos|> ROLE(m1) content(m1) <|end_turn|> .. ROLE(mk) content(mk) <|end_turn|> <|eos|>

where ``ROLE`` is ``<|system|>`` (3), ``<|user|>`` (4), ``<|assistant|>`` (5),
or ``<|tool|>`` (6) and ``<|end_turn|>`` is 7 (``epor.tokenizer.spec``).  Role
and turn tokens are inserted **by ID**; message text goes through the codec,
which refuses to produce a reserved ID from content, so no message can forge a
turn boundary or a role.

Loss mask
---------
``mask[i]`` says whether token ``i`` is a training *target* (predicted from
token ``i - 1``).  Only assistant content and the ``<|end_turn|>`` that closes
an assistant turn are targets, plus the closing ``<|eos|>`` when the last
message is the assistant's.  The ``<|assistant|>`` role token itself is not a
target: at generation time the prompt supplies it (:func:`encode_prompt`), so
the model only ever has to learn what follows it.

Generation prompts end with ``<|assistant|>``; generation stops at
``<|end_turn|>`` or ``<|eos|>``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from epor.data.store import canonical_json_bytes, sha256_bytes
from epor.tokenizer.spec import (
    MODEL_SEQUENCE_END_ID,
    MODEL_SEQUENCE_START_ID,
    RESERVED_SPEC_SHA256,
    SPECIAL_TOKEN_IDS,
)

# A conversation is a model sequence: it starts and ends exactly as the
# training kernel frames every document.
BOS_ID: Final = MODEL_SEQUENCE_START_ID
EOS_ID: Final = MODEL_SEQUENCE_END_ID

CHAT_FORMAT_VERSION: Final = "epor-chat-v1"
Role = Literal["system", "user", "assistant", "tool"]
ROLE_IDS: Final[dict[str, int]] = {
    "system": SPECIAL_TOKEN_IDS["<|system|>"],
    "user": SPECIAL_TOKEN_IDS["<|user|>"],
    "assistant": SPECIAL_TOKEN_IDS["<|assistant|>"],
    "tool": SPECIAL_TOKEN_IDS["<|tool|>"],
}
END_TURN_ID: Final = SPECIAL_TOKEN_IDS["<|end_turn|>"]
STOP_IDS: Final = (END_TURN_ID, EOS_ID)
MAX_MESSAGE_CHARS: Final = 16_384


def chat_format() -> dict[str, object]:
    """The hashable description of the format, pinned to the reserved-ID spec."""

    return {
        "version": CHAT_FORMAT_VERSION,
        "reserved_spec_sha256": RESERVED_SPEC_SHA256,
        "bos": BOS_ID,
        "eos": EOS_ID,
        "end_turn": END_TURN_ID,
        "roles": ROLE_IDS,
        "targets": "assistant content, its end_turn, and a final eos after an assistant turn",
    }


CHAT_FORMAT_SHA256: Final = sha256_bytes(canonical_json_bytes(chat_format()))


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Role
    content: str = Field(max_length=MAX_MESSAGE_CHARS)


class ContentTokenizer(Protocol):
    def encode(self, value: str) -> list[int]: ...


def _validate(messages: Sequence[ChatMessage], *, prompt: bool) -> None:
    if not messages:
        raise ValueError("a conversation needs at least one message")
    if not prompt and messages[-1].role != "assistant":
        raise ValueError("a training conversation must end with an assistant message")
    if not any(message.role == "assistant" for message in messages) and not prompt:
        raise ValueError("a training conversation needs an assistant message")


def encode_conversation(
    tokenizer: ContentTokenizer, messages: Sequence[ChatMessage]
) -> tuple[list[int], list[bool]]:
    """Token IDs and per-token target mask for one training conversation."""

    _validate(messages, prompt=False)
    ids = [BOS_ID]
    mask = [False]
    for message in messages:
        trained = message.role == "assistant"
        content = tokenizer.encode(message.content)
        ids.append(ROLE_IDS[message.role])
        mask.append(False)
        ids += content
        mask += [trained] * len(content)
        ids.append(END_TURN_ID)
        mask.append(trained)
    ids.append(EOS_ID)
    mask.append(True)
    return ids, mask


def encode_prompt(tokenizer: ContentTokenizer, messages: Sequence[ChatMessage]) -> list[int]:
    """Prompt IDs ending with ``<|assistant|>``, ready for generation."""

    _validate(messages, prompt=True)
    ids = [BOS_ID]
    for message in messages:
        ids.append(ROLE_IDS[message.role])
        ids += tokenizer.encode(message.content)
        ids.append(END_TURN_ID)
    ids.append(ROLE_IDS["assistant"])
    return ids


def encode_response(tokenizer: ContentTokenizer, response: str) -> list[int]:
    """The target IDs an assistant reply contributes after the prompt."""

    return [*tokenizer.encode(response), END_TURN_ID]


def messages_sha256(messages: Sequence[ChatMessage]) -> str:
    return sha256_bytes(canonical_json_bytes([message.model_dump() for message in messages]))


__all__ = [
    "CHAT_FORMAT_SHA256",
    "CHAT_FORMAT_VERSION",
    "END_TURN_ID",
    "ROLE_IDS",
    "STOP_IDS",
    "ChatMessage",
    "Role",
    "chat_format",
    "encode_conversation",
    "encode_prompt",
    "encode_response",
    "messages_sha256",
]
