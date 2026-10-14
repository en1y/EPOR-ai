"""Constitutional critique and revision that cites the ratified covenant by key.

**[disclosed]** Constitutional AI
(`anthropic-constitutional-ai <https://arxiv.org/abs/2212.08073>`_) has a
model critique its own response against a written principle and then revise
it.  **[EPOR-adaptation]** EPOR's "constitution" is ``configs/covenant-v1.yaml``
itself.  This module never restates a principle: it names obligations by key,
reads their text from the covenant file at run time through
:func:`epor.safety.load_covenant`, and records the covenant's SHA-256 and the
cited keys on every output.  ``tests/test_alignment_critique.py`` fails if
alignment code or configuration carries a second copy of covenant text.

Outputs are **candidates only**: each critique and revision is an
:class:`~epor.alignment.records.ExampleRecord` with ``candidate=True``.  A
revision becomes training data only through an AI-judged, human-audited
admission (:mod:`epor.alignment.audit`); a critique never does.  A revision
record's ``messages`` are the *original* prompt, so an admitted revision is a
demonstration for that prompt; the critique and revision prompts themselves
are fixed by the recorded template hash and the lineage (the original example,
the critique, and the covenant).

**[hypothesis]** Model feedback may reduce human exposure to disturbing
material without loss of quality.  It must beat SFT and human-preference
controls before it becomes a default; nothing here measures that yet.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from epor.data.store import canonical_json_bytes, sha256_bytes
from epor.safety import Covenant, load_covenant

from .chat import ChatMessage
from .records import (
    LOCAL_CHECKPOINT_RIGHTS,
    AlignmentRecordError,
    AlignmentStore,
    CovenantCitation,
    ExampleRecord,
    Lineage,
    Sampling,
    parent,
)

# Placeholders only.  The obligation text is substituted from the covenant
# file at run time and is never written into this module.
CRITIQUE_TEMPLATE = {
    "id": "epor-constitutional-critique-v1",
    "critique": (
        "Covenant obligation {key}: {statement}\n"
        "Request: {request}\nResponse: {response}\n"
        "Point out one way the response falls short of this obligation."
    ),
    "revision": (
        "Covenant obligation {key}: {statement}\n"
        "Request: {request}\nResponse: {response}\nCritique: {critique}\n"
        "Rewrite the response so that it meets this obligation."
    ),
}
CRITIQUE_TEMPLATE_SHA256 = sha256_bytes(canonical_json_bytes(CRITIQUE_TEMPLATE))


def covenant_obligations(covenant: Covenant | None = None) -> dict[str, tuple[int, str]]:
    """``{obligation key: (priority, statement)}`` read from the covenant file."""

    source = covenant or load_covenant()
    return {
        obligation.key: (principle.priority, obligation.statement)
        for principle in source.principles
        for obligation in principle.obligations
    }


def _request(messages: Sequence[ChatMessage]) -> str:
    return "\n".join(f"{message.role}: {message.content}" for message in messages)


def critique_and_revise(
    store: AlignmentStore,
    model: Any,
    example_ids: Sequence[str],
    obligation_keys: Sequence[str],
    sampling: Sampling,
    *,
    covenant: Covenant | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> list[dict[str, str]]:
    """One critique and one revision per (example, obligation); returns their record IDs."""

    source = covenant or load_covenant()
    obligations = covenant_obligations(source)
    unknown = sorted(set(obligation_keys) - set(obligations))
    if unknown:
        raise ValueError(f"not covenant obligation keys: {', '.join(unknown)}")
    if not obligation_keys:
        raise ValueError("cite at least one covenant obligation by key")
    covenant_lineage = Lineage(kind="covenant", id=source.covenant_id, sha256=source.sha256)
    written: list[dict[str, str]] = []
    for example_id in example_ids:
        example = store.get(example_id)
        if not isinstance(example, ExampleRecord) or example.purpose == "critique":
            raise AlignmentRecordError(
                "critique needs a demonstration, teacher output, or revision"
            )
        request = _request(example.messages)
        for key in obligation_keys:
            if should_cancel is not None and should_cancel():
                raise InterruptedError("critique cancelled")
            _, statement = obligations[key]
            citation = CovenantCitation(
                covenant_sha256=source.sha256,
                covenant_version=source.covenant_version,
                obligations=(key,),
            )
            common: dict[str, Any] = {
                "candidate": True,
                "generator": model.generator,
                "rights": LOCAL_CHECKPOINT_RIGHTS,
                "prompt_template_id": CRITIQUE_TEMPLATE["id"],
                "prompt_template_sha256": CRITIQUE_TEMPLATE_SHA256,
                "sampling": sampling,
                "covenant": citation,
            }
            critique_prompt = CRITIQUE_TEMPLATE["critique"].format(
                key=key, statement=statement, request=request, response=example.output
            )
            critique_text, critique_ids = model.generate(
                (ChatMessage(role="user", content=critique_prompt),), sampling
            )
            critique_id = store.put(
                ExampleRecord(
                    purpose="critique",
                    messages=example.messages,
                    output=critique_text,
                    output_token_ids=critique_ids,
                    created_at=datetime.now(UTC),
                    lineage=(parent(example_id), covenant_lineage, model.lineage()),
                    **common,
                )
            )
            revision_prompt = CRITIQUE_TEMPLATE["revision"].format(
                key=key,
                statement=statement,
                request=request,
                response=example.output,
                critique=critique_text,
            )
            revision_text, revision_ids = model.generate(
                (ChatMessage(role="user", content=revision_prompt),), sampling
            )
            revision_id = store.put(
                ExampleRecord(
                    purpose="revision",
                    messages=example.messages,
                    output=revision_text,
                    output_token_ids=revision_ids,
                    created_at=datetime.now(UTC),
                    lineage=(
                        parent(example_id),
                        parent(critique_id),
                        covenant_lineage,
                        model.lineage(),
                    ),
                    **common,
                )
            )
            written.append(
                {
                    "example": example_id,
                    "obligation": key,
                    "critique": critique_id,
                    "revision": revision_id,
                }
            )
    return written


__all__ = [
    "CRITIQUE_TEMPLATE",
    "CRITIQUE_TEMPLATE_SHA256",
    "covenant_obligations",
    "critique_and_revise",
]
