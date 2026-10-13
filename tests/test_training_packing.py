"""Sequence packing, token batching, exact resume, and rank partitioning (gate G3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from conftest import TokenShardFixture, make_token_shards  # noqa: E402
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from epor.models import EporDecoder, ModelConfig, load_model_config  # noqa: E402
from epor.models.reference import IGNORE_INDEX  # noqa: E402
from epor.tokenizer import BOS_ID, EOS_ID, PAD_ID, iter_tokenized_documents  # noqa: E402
from epor.training.packing import (  # noqa: E402
    DataPipelineSettings,
    PackedBatchStream,
    PackedStep,
    evaluation_steps,
)
from epor.training.shard_data import ShardCorpus  # noqa: E402

TINY_CONFIG = Path("configs/models/epor-tiny.yaml")


@pytest.fixture(scope="module")
def shards(tmp_path_factory: pytest.TempPathFactory) -> TokenShardFixture:
    return make_token_shards(
        tmp_path_factory.mktemp("packing"),
        extra_train_text="A short fourth document keeps the shuffle honest.\n",
    )


@pytest.fixture(scope="module")
def config(shards: TokenShardFixture) -> ModelConfig:
    base = load_model_config(TINY_CONFIG)
    return ModelConfig.model_validate(
        {**base.model_dump(mode="json"), "vocab_size": shards.vocab_size}
    )


@pytest.fixture(scope="module")
def train(shards: TokenShardFixture, config: ModelConfig) -> ShardCorpus:
    return ShardCorpus.open(shards.data_root, shards.tokenized_id, "train", model_config=config)


@pytest.fixture(scope="module")
def validation(shards: TokenShardFixture, config: ModelConfig) -> ShardCorpus:
    return ShardCorpus.open(
        shards.data_root, shards.tokenized_id, "validation", model_config=config
    )


def _stream(corpus: ShardCorpus, **overrides: Any) -> PackedBatchStream:
    options: dict[str, Any] = {"seq_len": 16, "tokens_per_batch": 64, "seed": 7, **overrides}
    return PackedBatchStream(corpus, **options)


def _assert_steps_equal(left: PackedStep, right: PackedStep) -> None:
    assert left.step == right.step
    assert left.global_target_tokens == right.global_target_tokens
    assert len(left.micro_batches) == len(right.micro_batches)
    for a, b in zip(left.micro_batches, right.micro_batches, strict=True):
        assert a.layout == b.layout
        assert torch.equal(a.input_ids, b.input_ids)
        assert torch.equal(a.targets, b.targets)
        assert torch.equal(a.segment_ids, b.segment_ids)


def _unpack(steps: list[PackedStep]) -> list[tuple[int, list[int]]]:
    """Rebuild ``(document, framed tokens)`` in stream order from packed rows."""

    documents: list[tuple[int, list[int]]] = []
    open_document: tuple[int, list[int]] | None = None
    for step in steps:
        for micro in step.micro_batches:
            for row_index, row in enumerate(micro.layout):
                position = 0
                for segment_index, segment in enumerate(row):
                    stop = position + segment.length
                    segment_ids = micro.segment_ids[row_index, position:stop]
                    assert segment_ids.eq(segment_index).all()
                    inputs = micro.input_ids[row_index, position:stop].tolist()
                    targets = micro.targets[row_index, position:stop].tolist()
                    assert inputs[1:] == targets[:-1]
                    if segment.start == 0:
                        assert open_document is None
                        open_document = (segment.document, [])
                    assert open_document is not None
                    assert open_document[0] == segment.document
                    assert len(open_document[1]) == segment.start
                    open_document[1].extend(inputs)
                    if segment.ends_document:
                        assert targets[-1] == EOS_ID
                        open_document[1].append(targets[-1])
                        documents.append(open_document)
                        open_document = None
                    else:
                        # A cut document is always the last run of its row.
                        assert segment_index == len(row) - 1 and stop == micro.input_ids.shape[1]
                        assert targets[-1] != EOS_ID
                    position = stop
                assert micro.segment_ids[row_index, position:].eq(len(row)).all()
    return documents


def test_packing_round_trips_documents_over_epochs(
    shards: TokenShardFixture, train: ShardCorpus
) -> None:
    stream = _stream(train, seq_len=24, tokens_per_batch=96)
    steps: list[PackedStep] = []
    while stream.epoch < 3:
        steps.append(stream.next_step())
    expected = {
        index: [BOS_ID, *content, EOS_ID]
        for index, (_, content) in enumerate(
            iter_tokenized_documents(shards.data_root, shards.tokenized_id, "train")
        )
    }
    documents = _unpack(steps)
    count = train.document_count
    assert count >= 4 and len(documents) >= 3 * count
    for epoch in range(3):
        visited = documents[epoch * count : (epoch + 1) * count]
        assert sorted(document for document, _ in visited) == list(range(count))
        for document, framed in visited:
            assert framed == expected[document]
    for step in steps:
        assert step.global_tokens == 96 and step.global_target_tokens == 96
        assert step.padding_fraction == 0.0
        for micro in step.micro_batches:
            assert micro.input_ids.shape == (4, 24)
            assert not micro.targets.eq(IGNORE_INDEX).any()
            assert not micro.input_ids.eq(EOS_ID).any()


def test_evaluation_pass_pads_only_the_tail(validation: ShardCorpus) -> None:
    steps = list(evaluation_steps(validation, seq_len=16, tokens_per_batch=48))
    documents = _unpack(steps)
    assert [document for document, _ in documents] == list(range(validation.document_count))
    real = sum(step.global_target_tokens for step in steps)
    assert real == validation.pair_count
    total = sum(step.global_tokens for step in steps)
    assert total == len(steps) * 48 and total - real < 48
    last = steps[-1].micro_batches[-1]
    padding = last.targets.eq(IGNORE_INDEX)
    assert int(padding.sum()) == last.padding_tokens > 0
    assert last.input_ids[padding].eq(PAD_ID).all()
    assert steps[-1].padding_fraction == pytest.approx(last.padding_tokens / 48)
    assert (
        len(list(evaluation_steps(validation, seq_len=16, tokens_per_batch=48, max_steps=1))) == 1
    )


def test_same_seed_is_identical_and_seed_changes_order(train: ShardCorpus) -> None:
    first, second = _stream(train), _stream(train)
    other = _stream(train, seed=8)
    layouts_first, layouts_other = [], []
    for _ in range(40):
        a, b, c = first.next_step(), second.next_step(), other.next_step()
        _assert_steps_equal(a, b)
        layouts_first.append(a.micro_batches[0].layout)
        layouts_other.append(c.micro_batches[0].layout)
    assert layouts_first != layouts_other


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    seq_len=st.integers(1, 40),
    rows=st.integers(1, 4),
    cut=st.integers(0, 30),
    seed=st.integers(0, 2**32),
)
def test_resume_reproduces_the_uninterrupted_stream(
    train: ShardCorpus, seq_len: int, rows: int, cut: int, seed: int
) -> None:
    options = {"seq_len": seq_len, "tokens_per_batch": seq_len * rows, "seed": seed}
    reference = _stream(train, **options)
    steps = [reference.next_step() for _ in range(cut + 4)]
    interrupted = _stream(train, **options)
    for _ in range(cut):
        interrupted.next_step()
    state = json.loads(json.dumps(interrupted.state_dict()))
    resumed = _stream(train, **options)
    resumed.load_state_dict(state)
    for expected in steps[cut:]:
        _assert_steps_equal(resumed.next_step(), expected)


def test_resume_cut_points_cover_mid_document_and_later_epochs(train: ShardCorpus) -> None:
    stream = _stream(train, seq_len=8, tokens_per_batch=8)
    states: list[dict[str, Any]] = []
    steps: list[PackedStep] = []
    while stream.epoch < 2:
        states.append(stream.state_dict())
        steps.append(stream.next_step())
    states.append(stream.state_dict())
    steps.extend(stream.next_step() for _ in range(3))
    assert any(state["offset"] > 0 for state in states)
    assert any(state["epoch"] == 1 and state["offset"] > 0 for state in states)
    assert states[-1]["epoch"] == 2
    for cut in range(0, len(states), 7):
        resumed = _stream(train, seq_len=8, tokens_per_batch=8)
        resumed.load_state_dict(states[cut])
        for expected in steps[cut : cut + 4]:
            _assert_steps_equal(resumed.next_step(), expected)


@pytest.mark.parametrize(("world_size", "micro"), [(1, 1), (2, 1), (2, 3), (3, 2), (6, 1)])
def test_rank_union_equals_the_single_process_global_batch(
    train: ShardCorpus, world_size: int, micro: int
) -> None:
    options = {"seq_len": 12, "tokens_per_batch": 72}  # six rows per step
    single = _stream(train, **options)
    ranks = [
        _stream(train, **options, rank=rank, world_size=world_size, micro_batch_size=micro)
        for rank in range(world_size)
    ]
    assert ranks[0].geometry.accumulation_steps == 6 // world_size // micro
    for _ in range(12):
        expected = single.next_step().micro_batches[0]
        parts = [stream.next_step() for stream in ranks]
        assert {part.global_target_tokens for part in parts} == {expected.target_tokens}
        micro_batches = [batch for part in parts for batch in part.micro_batches]
        assert all(batch.input_ids.shape[0] == micro for batch in micro_batches)
        assert torch.equal(torch.cat([b.input_ids for b in micro_batches]), expected.input_ids)
        assert torch.equal(torch.cat([b.targets for b in micro_batches]), expected.targets)
        assert torch.equal(torch.cat([b.segment_ids for b in micro_batches]), expected.segment_ids)
        assert sum(b.target_tokens for b in micro_batches) == expected.target_tokens
        # Every rank holds the same state, so any rank's checkpoint resumes all.
        assert all(stream.state_dict() == single.state_dict() for stream in ranks)


def test_rank_state_resumes_exactly_and_across_world_sizes(train: ShardCorpus) -> None:
    options = {"seq_len": 12, "tokens_per_batch": 48}
    single = _stream(train, **options)
    for _ in range(5):
        single.next_step()
    state = single.state_dict()
    expected = [single.next_step().micro_batches[0] for _ in range(3)]
    ranks = [_stream(train, **options, rank=rank, world_size=4) for rank in range(4)]
    for stream in ranks:
        stream.load_state_dict(state)
    for batch in expected:
        rows = torch.cat([stream.next_step().micro_batches[0].input_ids for stream in ranks])
        assert torch.equal(rows, batch.input_ids)


def test_evaluation_partition_matches_single_process(validation: ShardCorpus) -> None:
    single = list(evaluation_steps(validation, seq_len=8, tokens_per_batch=32))
    ranks = [
        list(evaluation_steps(validation, seq_len=8, tokens_per_batch=32, rank=r, world_size=2))
        for r in range(2)
    ]
    for index, step in enumerate(single):
        parts = [ranks[r][index].micro_batches[0].targets for r in range(2)]
        assert torch.equal(torch.cat(parts), step.micro_batches[0].targets)
        assert ranks[1][index].global_target_tokens == step.global_target_tokens


def test_resume_refuses_different_settings_or_unreproducible_state(train: ShardCorpus) -> None:
    stream = _stream(train)
    stream.next_step()
    state = stream.state_dict()
    for change in ({"seed": 8}, {"seq_len": 8}, {"tokens_per_batch": 128}):
        with pytest.raises(ValueError, match="settings differ"):
            _stream(train, **change).load_state_dict(state)
    for key, value in (
        ("order_sha256", "0" * 64),
        ("position", train.document_count),
        ("offset", -1),
        ("schema_version", 99),
    ):
        with pytest.raises(ValueError):
            _stream(train).load_state_dict({**state, key: value})
    # Rank layout is not part of the fingerprint; the global batch is unchanged.
    _stream(train, rank=1, world_size=2).load_state_dict(state)


def test_invalid_geometry_is_refused(train: ShardCorpus) -> None:
    with pytest.raises(ValueError, match="multiple of seq_len"):
        _stream(train, tokens_per_batch=60)
    with pytest.raises(ValueError, match="divisible by world_size"):
        _stream(train, tokens_per_batch=48, world_size=2)
    with pytest.raises(ValueError, match="micro_batch_size"):
        _stream(train, micro_batch_size=3)
    with pytest.raises(ValueError, match="rank"):
        _stream(train, rank=2, world_size=2)
    with pytest.raises(ValueError, match="context"):
        _stream(train, seq_len=256, tokens_per_batch=256)


def test_settings_are_a_stable_manifest_record(train: ShardCorpus) -> None:
    stream = _stream(train)
    record = stream.settings.model_dump(mode="json")
    assert record == {
        "pipeline_version": "epor-packing-v1",
        "tokenized_id": train.tokenized_id,
        "tokenizer_id": train.tokenizer_id,
        "split": "train",
        "seq_len": 16,
        "tokens_per_batch": 64,
        "seed": 7,
        "shuffle": True,
    }
    restored = DataPipelineSettings.model_validate(record)
    assert restored.fingerprint == stream.settings.fingerprint
    assert _stream(train, seed=8).settings.fingerprint != restored.fingerprint


def test_packed_batches_train_the_tiny_decoder(
    train: ShardCorpus, validation: ShardCorpus, config: ModelConfig
) -> None:
    model = EporDecoder(config, seed=3)
    whole = _stream(train, seq_len=32, tokens_per_batch=128).next_step()
    split = _stream(train, seq_len=32, tokens_per_batch=128, micro_batch_size=1).next_step()
    batch = whole.micro_batches[0]
    output = model(batch.input_ids, batch.targets, segment_ids=batch.segment_ids)
    assert output.loss is not None and torch.isfinite(output.loss)
    # Token-weighted micro-batch sums reproduce the global mean loss.
    total = torch.zeros(())
    for micro in split.micro_batches:
        part = model(micro.input_ids, micro.targets, segment_ids=micro.segment_ids)
        assert part.main_loss is not None
        total = total + part.main_loss * micro.target_tokens
    assert output.main_loss is not None
    torch.testing.assert_close(total / split.global_target_tokens, output.main_loss)
    tail = list(evaluation_steps(validation, seq_len=32, tokens_per_batch=64))[-1]
    padded = tail.micro_batches[0]
    assert padded.padding_tokens > 0
    evaluated = model(padded.input_ids, padded.targets, segment_ids=padded.segment_ids)
    assert evaluated.loss is not None and torch.isfinite(evaluated.loss)
