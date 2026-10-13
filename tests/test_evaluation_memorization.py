"""Extraction-rate and canary-exposure probes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import pytest
from conftest import TokenShardFixture, make_token_shards
from evaluation_support import TEST_HARNESS, StubScoringModel
from kernel_support import kernel_config, kernel_settings

from epor.evaluation import (
    CanarySettings,
    EvaluationCancelledError,
    MemorizationSettings,
    ScoringModel,
    SuiteDefinition,
    canary_exposure,
    generate_canary,
    run_memorization,
    run_suite,
)
from epor.evaluation.memorization import (
    DEFAULT_CANARY_TEMPLATE,
    canary_candidates,
    common_prefix_length,
    sample_windows,
)
from epor.evaluation.report import ModelIdentity, report_bytes
from epor.evaluation.scoring import Loglikelihood, truncate_at_stop

TRAIN_TEXTS = {
    f"train-{i}": f"Training document {i} talks about orchards, rivers, and ledgers in order {i}. "
    * 3
    for i in range(6)
}
HELDOUT_TEXTS = {
    f"held-{i}": f"Held-out note {i} discusses lanterns, bridges, and kettles at length {i}. " * 3
    for i in range(6)
}


class CharTokenizer:
    def encode(self, value: str) -> list[int]:
        return [ord(char) for char in value]

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(chr(token) for token in token_ids)


class MemorizingModel(ScoringModel):
    """Continues any prompt found verbatim in its 'training' texts; else emits nothing."""

    def __init__(self, texts: Sequence[str]) -> None:
        self.texts = list(texts)

    @property
    def identity(self) -> ModelIdentity:
        return ModelIdentity(
            kind="stub", model_id="memorizer", tokenizer_id="c", tokenizer_kind="c"
        )

    def loglikelihood(self, context: str, continuation: str) -> Loglikelihood:
        raise NotImplementedError

    def generate(self, prompt: str, *, max_new_tokens: int, stop: Sequence[str] = ()) -> str:
        for text in self.texts:
            index = text.find(prompt)
            if index >= 0:
                start = index + len(prompt)
                return truncate_at_stop(text[start : start + max_new_tokens], stop)[0]
        return ""


def _docs(texts: dict[str, str]) -> list[tuple[str, list[int]]]:
    tokenizer = CharTokenizer()
    return [(key, tokenizer.encode(value)) for key, value in texts.items()]


SETTINGS = MemorizationSettings(prefix_tokens=24, suffix_tokens=16, samples=4)


def test_memorizing_stub_extracts_train_text_but_not_held_out() -> None:
    model = MemorizingModel(list(TRAIN_TEXTS.values()))
    result = run_memorization(
        "mem",
        model,
        CharTokenizer(),
        _docs(TRAIN_TEXTS),
        _docs(HELDOUT_TEXTS),
        scope="harness-fixture",
        settings=SETTINGS,
        seed=3,
    )
    assert result.metric("train.exact_suffix_rate").value == 1.0
    assert result.metric("heldout.exact_suffix_rate").value == 0.0
    assert result.metric("extraction_gap").value == 1.0
    assert result.metric("train.mean_lcp_fraction").value == 1.0
    assert result.metric("train.samples").value == 4.0
    assert result.metric("heldout.samples").value == 4.0
    # Reports name documents and offsets, never their text.
    payload = result.model_dump_json()
    assert "orchards" not in payload and "lanterns" not in payload
    assert {item.details["split"] for item in result.items} == {"train", "validation"}


def test_sampling_is_deterministic_and_seeded() -> None:
    docs = _docs({**TRAIN_TEXTS, **{f"x{i}": "short" for i in range(3)}})

    def sample(seed: int) -> list[tuple[str, int]]:
        windows = sample_windows(
            docs, split="train", prefix_tokens=24, suffix_tokens=16, count=3, seed=seed
        )
        return [(w.document_id, w.offset) for w in windows]

    assert sample(1) == sample(1)
    assert sample(1) != sample(2)
    assert len(sample(1)) == 3
    assert all(not doc.startswith("x") for doc, _ in sample(1))  # too short to sample
    everything = sample_windows(
        docs, split="train", prefix_tokens=24, suffix_tokens=16, count=100, seed=1
    )
    assert len(everything) == len(TRAIN_TEXTS)


def test_common_prefix_length() -> None:
    assert common_prefix_length("abcdef", "abcxyz") == 3
    assert common_prefix_length("", "abc") == 0


def test_memorization_cancels() -> None:
    with pytest.raises(EvaluationCancelledError):
        run_memorization(
            "mem",
            MemorizingModel([]),
            CharTokenizer(),
            _docs(TRAIN_TEXTS),
            _docs(HELDOUT_TEXTS),
            scope="harness-fixture",
            settings=SETTINGS,
            should_cancel=lambda: True,
        )


# ------------------------------------------------------------------------- canaries


class CanaryModel(StubScoringModel):
    """Candidates score by a fixed hash-derived value; the canary by ``boost``."""

    def __init__(self, canary_secret: str, boost: float) -> None:
        super().__init__()
        self.canary_secret = canary_secret
        self.boost = boost

    def loglikelihood(self, context: str, continuation: str) -> Loglikelihood:
        if continuation == self.canary_secret:
            return Loglikelihood(sum_logprob=self.boost, is_greedy=False)
        return Loglikelihood(sum_logprob=-(int(continuation) % 997) / 10.0, is_greedy=False)


def test_canaries_are_deterministic_and_high_entropy() -> None:
    first = generate_canary(5, 0)
    assert first == generate_canary(5, 0)
    assert first != generate_canary(5, 1)
    assert first != generate_canary(6, 0)
    assert first.startswith("The canary code is ")
    candidates = canary_candidates(first, 50, seed=5)
    assert len(set(candidates)) == 50 and first not in candidates
    with pytest.raises(ValueError, match="entropy"):
        CanarySettings(length=4, alphabet="ab", candidates=100)


def test_canary_exposure_is_monotone_in_the_canary_likelihood() -> None:
    canary = generate_canary(0, 0)
    secret = canary.removeprefix(DEFAULT_CANARY_TEMPLATE.removesuffix("{secret}"))
    candidates = canary_candidates(canary, 127, seed=0)
    exposures = [
        canary_exposure(CanaryModel(secret, boost), canary, candidates).exposure_bits
        for boost in (-200.0, -80.0, -50.0, -20.0, -5.0, 0.5)
    ]
    assert exposures == sorted(exposures)
    assert exposures[0] == 0.0  # least likely: rank 128 of 128
    assert exposures[-1] == pytest.approx(math.log2(128))  # most likely: rank 1


def test_canary_task_in_a_suite_is_a_baseline() -> None:
    suite = SuiteDefinition.model_validate(
        {
            "id": "canary-suite",
            "version": 1,
            "scope": "harness-fixture",
            "title": "Canary",
            "tasks": [{"id": "canaries", "type": "canary-exposure", "canary": {"count": 2}}],
        }
    )
    report = run_suite(suite, StubScoringModel(default_logprob=-3.0), harness=TEST_HARNESS)
    task = report.task("canaries")
    # All ties count against the canary: no exposure.
    assert task.metric("mean_exposure_bits").value == 0.0
    assert task.settings["injected"] is False
    assert generate_canary(0, 0) not in report_bytes(report).decode()
    again = run_suite(suite, StubScoringModel(default_logprob=-3.0), harness=TEST_HARNESS)
    assert again.report_id == report.report_id


# ------------------------------------------------------------------ real tiny model

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def shards(tmp_path_factory: pytest.TempPathFactory) -> TokenShardFixture:
    return make_token_shards(tmp_path_factory.mktemp("memorization"))


def test_overfit_tiny_model_extracts_train_at_least_as_well(
    shards: TokenShardFixture, tmp_path: Path
) -> None:
    from epor.evaluation.decoder import load_kernel_scoring_model
    from epor.training import OptimizerSettings, pretrain_packed

    config = kernel_config(shards.vocab_size)
    settings = kernel_settings(
        shards.tokenized_id,
        checkpoint_every=40,
        optimizer=OptimizerSettings(learning_rate=1e-2, warmup_steps=2, total_steps=40),
    )
    run = tmp_path / "run"
    pretrain_packed(config, shards.data_root, run, settings, max_steps=40)
    model = load_kernel_scoring_model(run, shards.data_root)
    suite = SuiteDefinition.model_validate(
        {
            "id": "memorization-suite",
            "version": 1,
            "scope": "proxy-scale",
            "title": "Memorization",
            "seed": 7,
            "tasks": [
                {
                    "id": "extraction",
                    "type": "memorization",
                    "memorization": {"prefix_tokens": 8, "suffix_tokens": 8, "samples": 6},
                }
            ],
        }
    )
    report = run_suite(suite, model, harness=TEST_HARNESS)
    task = report.task("extraction")
    train = task.metric("train.mean_lcp_fraction").value
    heldout = task.metric("heldout.mean_lcp_fraction").value
    assert train is not None and heldout is not None
    assert train >= heldout
    assert (task.metric("train.exact_suffix_rate").value or 0.0) >= (
        task.metric("heldout.exact_suffix_rate").value or 0.0
    )
    assert task.settings["tokenized_id"] == shards.tokenized_id
    assert [s.source_id for s in report.data if s.kind == "tokenized-build"] == [
        shards.tokenized_id
    ]
    again = run_suite(suite, model, harness=TEST_HARNESS)
    assert again.report_id == report.report_id


def test_memorization_skips_without_training_data() -> None:
    suite = SuiteDefinition.model_validate(
        {
            "id": "memorization-suite",
            "version": 1,
            "scope": "harness-fixture",
            "title": "Memorization",
            "tasks": [
                {"id": "extraction", "type": "memorization"},
                {"id": "canaries", "type": "canary-exposure"},
            ],
        }
    )
    report = run_suite(suite, StubScoringModel(default_logprob=-1.0), harness=TEST_HARNESS)
    assert [(s.task_id, s.reason) for s in report.skipped_tasks] == [
        ("extraction", "no training data (tokenized build) was supplied")
    ]
