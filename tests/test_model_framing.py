"""Every kernel-model consumer frames sequences exactly as the training kernel does."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from conftest import make_token_shards  # noqa: E402
from kernel_support import kernel_config, kernel_settings  # noqa: E402

from epor.alignment.chat import (  # noqa: E402
    STOP_IDS,
    ChatMessage,
    encode_conversation,
    encode_prompt,
)
from epor.alignment.model import load_chat_model  # noqa: E402
from epor.evaluation.decoder import load_kernel_scoring_model  # noqa: E402
from epor.tokenizer.spec import DOC_END_ID, DOC_START_ID  # noqa: E402
from epor.training import pretrain_packed  # noqa: E402
from epor.training.packing import PackedBatchStream  # noqa: E402
from epor.training.shard_data import ShardCorpus  # noqa: E402


def test_scoring_chat_and_training_agree_on_the_sequence_frame(tmp_path: Path) -> None:
    shards = make_token_shards(tmp_path)
    run = tmp_path / "run"
    settings = kernel_settings(shards.tokenized_id)
    pretrain_packed(kernel_config(shards.vocab_size), shards.data_root, run, settings, max_steps=1)

    corpus = ShardCorpus.open(shards.data_root, shards.tokenized_id, "train")
    inputs, targets = corpus.pairs(0, 0, corpus.pair_counts[0])
    trained_start = int(inputs[0])
    trained_end = int(targets[-1])
    stream = PackedBatchStream(corpus, seq_len=16, tokens_per_batch=32, seed=0)
    first_row = stream.next_step().micro_batches[0].input_ids[0]
    assert int(first_row[0]) == trained_start  # what the model sees first, every document
    every_input = torch.cat([corpus.pairs(d, 0, corpus.pair_counts[d])[0] for d in range(2)])
    every_target = torch.cat([corpus.pairs(d, 0, corpus.pair_counts[d])[1] for d in range(2)])
    assert DOC_START_ID not in every_input.tolist() and DOC_END_ID not in every_target.tolist()

    scoring = load_kernel_scoring_model(run, shards.data_root)
    assert scoring.prefix_ids == [trained_start]
    assert scoring.stop_ids == [trained_end]

    chat = load_chat_model(run, shards.data_root)
    messages = (ChatMessage(role="user", content="hi"),)
    assert encode_prompt(chat.tokenizer, messages)[0] == trained_start
    ids, _ = encode_conversation(
        chat.tokenizer, (*messages, ChatMessage(role="assistant", content="ok"))
    )
    assert (ids[0], ids[-1]) == (trained_start, trained_end)
    assert trained_end in STOP_IDS and DOC_END_ID not in STOP_IDS
