"""Domain-weighted training subsets: selection, hash neutrality, and kernel integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from conftest import TokenShardFixture, make_domain_shards  # noqa: E402
from kernel_support import kernel_config, kernel_settings  # noqa: E402

from epor.training import (  # noqa: E402
    DomainMixture,
    ShardCorpus,
    evaluate_packed_checkpoint,
    pretrain_packed,
    select_mixture_documents,
)
from epor.training.kernel import PackedTrainingSettings  # noqa: E402
from epor.training.mixture import document_domains  # noqa: E402


@pytest.fixture(scope="module")
def mixed_build(tmp_path_factory: pytest.TempPathFactory) -> TokenShardFixture:
    return make_domain_shards(tmp_path_factory.mktemp("mixture"))


def test_mixture_selection_is_deterministic_and_matches_weights(
    mixed_build: TokenShardFixture,
) -> None:
    mixture = DomainMixture(weights={"code": 1.0, "general": 1.0}, seed=3)
    first = select_mixture_documents(mixed_build.data_root, mixed_build.tokenized_id, mixture)
    second = select_mixture_documents(mixed_build.data_root, mixed_build.tokenized_id, mixture)
    assert first == second
    shares = {item.domain: item for item in first.domains}
    assert set(shares) == {"code", "general"}
    for item in shares.values():
        assert item.target_share == 0.5
        assert 0 < item.selected_documents <= item.available_documents
    # The scarcer domain is taken whole; the other overshoots by less than one document.
    assert any(item.selected_pairs == item.available_pairs for item in shares.values())

    only_code = select_mixture_documents(
        mixed_build.data_root,
        mixed_build.tokenized_id,
        DomainMixture(weights={"code": 1.0, "general": 0.0}),
    )
    code = {item.domain: item for item in only_code.domains}
    assert code["general"].selected_documents == 0
    assert code["code"].realized_share == 1.0
    with pytest.raises(ValueError, match="absent"):
        select_mixture_documents(
            mixed_build.data_root,
            mixed_build.tokenized_id,
            DomainMixture(weights={"math-science": 1.0}),
        )


def test_untouched_trajectory_hash_and_mixture_resume_guard(
    mixed_build: TokenShardFixture,
) -> None:
    plain = kernel_settings(mixed_build.tokenized_id)
    assert "train_mixture" not in plain.trajectory()
    mixed = PackedTrainingSettings.model_validate(
        {**plain.model_dump(mode="json"), "train_mixture": {"weights": {"code": 1.0}}}
    )
    assert mixed.trajectory()["train_mixture"] == {"weights": {"code": 1.0}, "seed": 0}
    assert mixed.trajectory_sha256 != plain.trajectory_sha256


def test_corpus_selection_keeps_split_order_and_refuses_bad_ids(
    mixed_build: TokenShardFixture,
) -> None:
    corpus = ShardCorpus.open(mixed_build.data_root, mixed_build.tokenized_id, "train")
    chosen = [corpus.document_ids[2], corpus.document_ids[0]]
    view = corpus.select(chosen)
    assert view.document_ids == (corpus.document_ids[0], corpus.document_ids[2])
    assert view.pair_counts == (corpus.pair_counts[0], corpus.pair_counts[2])
    assert torch.equal(view.framed_tokens(1), corpus.framed_tokens(2))
    with pytest.raises(ValueError, match="repeat"):
        corpus.select([chosen[0], chosen[0]])
    with pytest.raises(ValueError, match="not in this split"):
        corpus.select(["0" * 64])


def test_kernel_trains_on_the_mixture_subset_and_records_it(
    mixed_build: TokenShardFixture, tmp_path: Path
) -> None:
    mixture = DomainMixture(weights={"code": 1.0, "general": 0.0})
    settings = PackedTrainingSettings.model_validate(
        {
            **kernel_settings(mixed_build.tokenized_id).model_dump(mode="json"),
            "train_mixture": mixture.model_dump(mode="json"),
        }
    )
    config = kernel_config(mixed_build.vocab_size)
    result = pretrain_packed(config, mixed_build.data_root, tmp_path / "run", settings, max_steps=2)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    selection = select_mixture_documents(mixed_build.data_root, mixed_build.tokenized_id, mixture)
    assert manifest["train_mixture"] == selection.summary()
    assert manifest["train_mixture"]["selected_documents"] == 4

    corpus = ShardCorpus.open(mixed_build.data_root, mixed_build.tokenized_id, "validation")
    domains = document_domains(
        mixed_build.data_root, corpus.manifest.dataset_id, corpus.manifest.build_id
    )
    code_ids = [doc for doc in corpus.document_ids if domains[doc] == "code"]
    whole = evaluate_packed_checkpoint(result.checkpoint_path, mixed_build.data_root)
    code = evaluate_packed_checkpoint(
        result.checkpoint_path, mixed_build.data_root, document_ids=code_ids
    )
    assert 0 < code.evaluation.target_tokens < whole.evaluation.target_tokens

    # A resume must keep the mixture: dropping it changes the trajectory.
    from epor.training.run_checkpoint import CheckpointError

    with pytest.raises(CheckpointError, match="train_mixture"):
        pretrain_packed(
            config,
            mixed_build.data_root,
            tmp_path / "run",
            settings.model_copy(update={"train_mixture": None}),
            max_steps=3,
            resume_from=tmp_path / "run",
        )
