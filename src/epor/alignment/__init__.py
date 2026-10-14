"""Knowledge transfer and alignment (v0.0.7).

Modules, in dependency order:

``chat``        the ``epor-chat-v1`` format on the frozen reserved role tokens
``records``     teacher-data provenance: immutable records and removal propagation
``fixtures``    project-authored demonstrations, preference pairs, and prompts
``model``       a verified kernel checkpoint driven through the chat format
``teachers``    EPOR-checkpoint sampling and operator-registered teacher outputs
``datasets``    content-addressed SFT and preference datasets with admission rules

SFT and logit distillation train inside the v0.0.5 kernel
(:func:`epor.training.pretrain_packed`); see :mod:`epor.training.kernel`.

Nothing in this package is evidence of alignment or safety.  It is a pipeline
whose every artifact is labelled, traceable, and reviewable by a human, run at
proxy scale on a CPU.
"""

from __future__ import annotations

from .chat import (
    CHAT_FORMAT_SHA256,
    CHAT_FORMAT_VERSION,
    ChatMessage,
    encode_conversation,
    encode_prompt,
    encode_response,
)
from .records import (
    AlignmentRecordError,
    AlignmentStore,
    ExampleRecord,
    Generator,
    JudgementRecord,
    Lineage,
    PreferenceRecord,
    Rights,
    Sampling,
    TeacherSource,
)

__all__ = [
    "CHAT_FORMAT_SHA256",
    "CHAT_FORMAT_VERSION",
    "AlignmentRecordError",
    "AlignmentStore",
    "ChatMessage",
    "ExampleRecord",
    "Generator",
    "JudgementRecord",
    "Lineage",
    "PreferenceRecord",
    "Rights",
    "Sampling",
    "TeacherSource",
    "encode_conversation",
    "encode_prompt",
    "encode_response",
]
