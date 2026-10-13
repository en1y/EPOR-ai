from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from epor.control.api import create_app
from epor.control.authority import Actor, AuthorityStore
from epor.control.settings import ControlSettings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class Harness:
    """A loopback control app with a bootstrapped owner, isolated per test.

    ``safety_root`` never points at the project's own ``.epor``: a fixture must
    not read, overwrite, or authenticate against the developer's real owner
    credential.
    """

    settings: ControlSettings
    app: Any
    store: AuthorityStore
    owner_token: str
    owner: Actor

    def client(self, **kwargs: Any) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            base_url="http://control.test",
            **kwargs,
        )

    def as_owner(self) -> httpx.AsyncClient:
        return self.client(headers={"Authorization": f"Bearer {self.owner_token}"})

    def as_bearer(self, token: str) -> httpx.AsyncClient:
        return self.client(headers={"Authorization": f"Bearer {token}"})

    def delegate(self, *scopes: str, days: int = 1, label: str = "operator") -> str:
        issued = self.store.create_delegation(
            label=label,
            scopes=list(scopes),
            expires_at=datetime.now(UTC) + timedelta(days=days),
            by=self.owner,
        )
        return issued.token


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    settings = ControlSettings(
        project_root=PROJECT_ROOT,
        database_path=tmp_path / "control.sqlite3",
        artifact_root=tmp_path / "artifacts",
        safety_root=tmp_path / "safety",
        data_root=tmp_path / "data",
        research_catalog_path=PROJECT_ROOT / "research" / "catalog.yaml",
        poll_interval_seconds=0.005,
    )
    settings.prepare_directories()
    assert settings.safety_root is not None
    store = AuthorityStore(settings.safety_root)
    issued = store.bootstrap_owner()
    return Harness(
        settings=settings,
        app=create_app(settings),
        store=store,
        owner_token=issued.token,
        owner=issued.principal.actor(),
    )


def run(scenario: Callable[[], Coroutine[Any, Any, None]]) -> None:
    """Drive one async API scenario from a synchronous test."""

    asyncio.run(scenario())


# --- Tokenizer fixtures -------------------------------------------------------

TOKENIZER_STAMP = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
TOKENIZER_SPLIT_SALT = "epor-data-split-v1"
TOKENIZER_FIXTURE_VOCAB = 1_024
# A string that appears nowhere in the train corpus; it must leave no trace in a
# tokenizer trained while its document sits in the validation split.
VALIDATION_MARKER = "qzvxjwkpf"

TOKENIZER_TRAIN_TEXTS: dict[str, str] = {
    "prose.txt": (
        "The quick brown fox jumps over the lazy dog while the archive records each "
        "change. Every document keeps its provenance, and the system rebuilds the same "
        "manifest from the same inputs. "
    )
    * 12,
    "module.py": (
        "def tokenize(value: str) -> list[int]:\n"
        "\tif not value:\n"
        "\t\treturn []  \n"
        "    result = [ord(item) for item in value]\n"
        "    return result\n\n"
        "class Encoder:\n"
        "    def __init__(self, size: int) -> None:\n"
        "        self.size = size\n"
    )
    * 10,
    "notes.md": (
        "# Notes\n\n- Unicode: café, naïve, Ελληνικά, 日本語, emoji 🙂🚀\n"
        "- Whitespace:   three spaces, a\ttab, and trailing space \n"
        "- Brackets <|like|> these and <tags> appear in prose too.\n\n"
    )
    * 10,
}


def group_for_split(split: str, *, train: int = 50, validation: int = 50) -> str:
    """Find a deterministic content-family ID that the data engine assigns to ``split``."""

    from epor.data.pipeline import split_for

    for index in range(10_000):
        candidate = f"family-{split}-{index}"
        if split_for(candidate, TOKENIZER_SPLIT_SALT, train, validation) == split:
            return candidate
    raise AssertionError(f"no group found for split {split}")


def tokenizer_registration(source_id: str, group_id: str) -> Any:
    from epor.data.models import SourceRegistration

    return SourceRegistration.model_validate(
        {
            "id": source_id,
            "title": f"Tokenizer fixture {source_id}",
            "owner_or_steward": "EPOR test suite",
            "canonical_origin": f"fixture://{source_id}",
            "acquisition_method": "generated_fixture",
            "acquired_at": TOKENIZER_STAMP.isoformat(),
            "media_type": "text/plain",
            "license_id": "fixture-only",
            "permission_basis": "Generated locally for deterministic tests.",
            "allowed_uses": ["research", "train", "evaluate"],
            "sensitive_content_risk": "low",
            "removal_contact": "owner@example.invalid",
            "group_id": group_id,
        }
    )


def make_tokenizer_build(
    root: Path,
    *,
    marker_split: str | None = "validation",
    extra_train_text: str | None = None,
) -> Any:
    """Ingest the fixture corpus and build it; returns the ``DatasetBuild``."""

    from epor.data.models import BuildRequest, SplitRatios
    from epor.data.service import DataEngine

    engine = DataEngine(root / "data")
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    texts = dict(TOKENIZER_TRAIN_TEXTS)
    if extra_train_text is not None:
        texts["extra.txt"] = extra_train_text
    paths = []
    for name, text in texts.items():
        path = inputs / name
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    engine.ingest(tokenizer_registration("train-corpus", group_for_split("train")), paths)
    if marker_split is not None:
        marker = inputs / "marker.txt"
        marker.write_text(
            " ".join([VALIDATION_MARKER] * 80) + "\nheld out evaluation text only\n",
            encoding="utf-8",
        )
        engine.ingest(
            tokenizer_registration("marker-corpus", group_for_split(marker_split)), [marker]
        )
    build, _ = engine.build(
        BuildRequest(
            dataset_id="tokenizer-fixture",
            split_ratios=SplitRatios(train=50, validation=50, test=0),
            split_salt=TOKENIZER_SPLIT_SALT,
            intended_uses=["tokenizer tests"],
            prohibited_uses=["production deployment"],
        )
    )
    return build


@pytest.fixture(scope="session")
def trained_tokenizer(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """One fixture tokenizer shared by read-only codec tests."""

    from epor.tokenizer import load_tokenizer, train_tokenizer

    root = tmp_path_factory.mktemp("tokenizer")
    build = make_tokenizer_build(root)
    trained = train_tokenizer(
        root / "data", build.dataset_id, build.build_id, vocab_size=TOKENIZER_FIXTURE_VOCAB
    )
    return load_tokenizer(root / "data", trained.tokenizer_id)


@dataclass(frozen=True, slots=True)
class TokenShardFixture:
    """A tiny real tokenized build: data root, IDs, and the build's vocabulary."""

    data_root: Path
    tokenized_id: str
    tokenizer_id: str
    vocab_size: int


def make_token_shards(
    root: Path, *, max_shard_tokens: int = 64, extra_train_text: str | None = None
) -> TokenShardFixture:
    """Build, train a tokenizer on, and shard the fixture corpus under ``root``.

    The small default shard limit forces multi-part documents.
    """

    from epor.tokenizer import build_token_shards, train_tokenizer

    build = make_tokenizer_build(root, extra_train_text=extra_train_text)
    data = root / "data"
    trained = train_tokenizer(
        data, build.dataset_id, build.build_id, vocab_size=TOKENIZER_FIXTURE_VOCAB
    )
    shards = build_token_shards(
        data,
        build.dataset_id,
        build.build_id,
        trained.tokenizer_id,
        max_shard_tokens=max_shard_tokens,
    )
    return TokenShardFixture(
        data_root=data,
        tokenized_id=shards.tokenized_id,
        tokenizer_id=trained.tokenizer_id,
        vocab_size=shards.manifest.vocab_size,
    )
