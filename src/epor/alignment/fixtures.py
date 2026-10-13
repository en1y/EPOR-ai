"""Project-authored alignment fixtures (``configs/alignment/fixtures.yaml``) as records."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, model_validator

from epor.data.models import FrozenSchema, Identifier
from epor.data.store import canonical_json_bytes, sha256_bytes

from .chat import ChatMessage
from .records import (
    AUTHORED,
    PROJECT_RIGHTS,
    AlignmentStore,
    ExampleRecord,
    Generator,
    Lineage,
    PreferenceRecord,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_PATH = PROJECT_ROOT / "configs" / "alignment" / "fixtures.yaml"
Text = Annotated[str, Field(min_length=1, max_length=2_000)]
PROJECT_AUTHOR = Generator(
    kind="project-authored", provider="EPOR project", model="authored fixtures", version="1"
)
FIXTURE_TEMPLATE = {"id": "epor-fixture-authored-v1", "kind": "authored demonstration"}
FIXTURE_TEMPLATE_SHA256 = sha256_bytes(canonical_json_bytes(FIXTURE_TEMPLATE))


class FixtureDemonstration(FrozenSchema):
    id: Identifier
    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    response: Text


class FixturePreference(FrozenSchema):
    id: Identifier
    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    chosen: Text
    rejected: Text


class FixturePrompt(FrozenSchema):
    id: Identifier
    messages: tuple[ChatMessage, ...] = Field(min_length=1)


class AlignmentFixtures(FrozenSchema):
    schema_version: Literal[1]
    id: Identifier
    version: int = Field(ge=1)
    demonstrations: tuple[FixtureDemonstration, ...]
    preferences: tuple[FixturePreference, ...]
    prompts: tuple[FixturePrompt, ...]

    @model_validator(mode="after")
    def unique_ids(self) -> AlignmentFixtures:
        ids = [
            item.id
            for group in (self.demonstrations, self.preferences, self.prompts)
            for item in group
        ]
        if len(ids) != len(set(ids)):
            raise ValueError("fixture IDs must be unique across sections")
        return self


def load_fixtures(path: Path = FIXTURES_PATH) -> AlignmentFixtures:
    return AlignmentFixtures.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _lineage(fixtures: AlignmentFixtures, item_id: str, item: FrozenSchema) -> Lineage:
    return Lineage(
        kind="project-fixture",
        id=f"{fixtures.id}/v{fixtures.version}/{item_id}",
        sha256=sha256_bytes(canonical_json_bytes(item)),
    )


def demonstration_records(fixtures: AlignmentFixtures) -> list[ExampleRecord]:
    return [
        ExampleRecord(
            purpose="demonstration",
            candidate=False,
            generator=PROJECT_AUTHOR,
            rights=PROJECT_RIGHTS,
            prompt_template_id=FIXTURE_TEMPLATE["id"],
            prompt_template_sha256=FIXTURE_TEMPLATE_SHA256,
            sampling=AUTHORED,
            messages=item.messages,
            output=item.response,
            lineage=(_lineage(fixtures, item.id, item),),
        )
        for item in fixtures.demonstrations
    ]


def preference_records(fixtures: AlignmentFixtures) -> list[PreferenceRecord]:
    return [
        PreferenceRecord(
            origin="project-fixture",
            messages=item.messages,
            chosen=item.chosen,
            rejected=item.rejected,
            lineage=(_lineage(fixtures, item.id, item),),
        )
        for item in fixtures.preferences
    ]


def prompt_lineage(fixtures: AlignmentFixtures, prompt: FixturePrompt) -> Lineage:
    return _lineage(fixtures, prompt.id, prompt)


def seed_fixtures(store: AlignmentStore, path: Path = FIXTURES_PATH) -> dict[str, list[str]]:
    """Write every fixture demonstration and preference as an immutable record.

    Fixture records carry no timestamp, so seeding is idempotent: the same
    file always yields the same record IDs.
    """

    fixtures = load_fixtures(path)
    return {
        "examples": [store.put(record) for record in demonstration_records(fixtures)],
        "preferences": [store.put(record) for record in preference_records(fixtures)],
    }


__all__ = [
    "FIXTURES_PATH",
    "PROJECT_AUTHOR",
    "AlignmentFixtures",
    "FixturePrompt",
    "demonstration_records",
    "load_fixtures",
    "preference_records",
    "prompt_lineage",
    "seed_fixtures",
]
