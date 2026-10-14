"""Teachers, response and logit distillation, and the DPO baseline."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

torch = pytest.importorskip("torch")

from alignment_support import (  # noqa: E402
    AlignmentFixture,
    checkpoint_ref,
    make_alignment_fixture,
    sft_settings,
)
from conftest import tokenizer_registration  # noqa: E402
from kernel_support import kernel_config, kernel_settings, same_weights  # noqa: E402

from epor.alignment.chat import ChatMessage  # noqa: E402
from epor.alignment.datasets import build_preference_dataset, build_sft_dataset  # noqa: E402
from epor.alignment.fixtures import load_fixtures, prompt_lineage, seed_fixtures  # noqa: E402
from epor.alignment.model import load_chat_model  # noqa: E402
from epor.alignment.preference import (  # noqa: E402
    DpoMonitor,
    DpoSettings,
    MonitorAbortError,
    dpo_terms,
    train_preference,
)
from epor.alignment.records import (  # noqa: E402
    PreferenceRecord,
    Sampling,
    parent,
)
from epor.alignment.teachers import (  # noqa: E402
    TeacherRefusedError,
    import_teacher_outputs,
    register_teacher,
    sample_teacher,
)
from epor.data.service import DataEngine  # noqa: E402
from epor.training import OptimizerSettings, TrainingCancelled, pretrain_packed  # noqa: E402
from epor.training.kernel import (  # noqa: E402
    DistillationRefusedError,
    LogitDistillation,
    distillation_kl_sum,
    final_model_state,
)
from epor.training.run_checkpoint import CheckpointError, latest_checkpoint  # noqa: E402
from epor.training.runner import resume_inputs  # noqa: E402


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> AlignmentFixture:
    return make_alignment_fixture(tmp_path_factory.mktemp("alignment-transfer"))


@pytest.fixture(scope="module")
def seeded(fixture: AlignmentFixture) -> dict[str, list[str]]:
    return seed_fixtures(fixture.store)


@pytest.fixture(scope="module")
def sft_checkpoint(
    fixture: AlignmentFixture,
    seeded: dict[str, list[str]],
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    dataset_id, _ = build_sft_dataset(fixture.store, fixture.tokenizer, seeded["examples"])
    run = tmp_path_factory.mktemp("sft") / "run"
    pretrain_packed(
        fixture.config,
        fixture.shards.data_root,
        run,
        sft_settings(fixture, dataset_id, validation_split=None),
        max_steps=2,
    )
    directory = latest_checkpoint(run)
    assert directory is not None
    return directory


@pytest.fixture(scope="module")
def preference_dataset(fixture: AlignmentFixture, seeded: dict[str, list[str]]) -> str:
    dataset_id, _ = build_preference_dataset(
        fixture.store, fixture.tokenizer, seeded["preferences"], validation_percent=35
    )
    return dataset_id


def _prompts(count: int = 3) -> list[tuple[tuple[ChatMessage, ...], tuple[Any, ...]]]:
    fixtures = load_fixtures()
    return [(item.messages, (prompt_lineage(fixtures, item),)) for item in fixtures.prompts[:count]]


# --- teachers -----------------------------------------------------------------


def test_epor_teacher_samples_record_exact_outputs_identity_and_lineage(
    fixture: AlignmentFixture,
) -> None:
    teacher = load_chat_model(fixture.base_checkpoint, fixture.shards.data_root)
    sampling = Sampling(
        decoding="temperature", temperature=0.8, top_k=20, max_new_tokens=12, seed=5
    )
    first = sample_teacher(fixture.store, teacher, _prompts(), sampling)
    again = [fixture.store.get(item) for item in first]
    for record in again:
        assert record.generator.kind == "epor-checkpoint"
        assert record.generator.checkpoint_sha256 == fixture.base_ref.checkpoint_sha256
        assert record.generator.tokenizer_id == fixture.shards.tokenizer_id
        assert record.output == fixture.tokenizer.decode(list(record.output_token_ids))
        assert all(token >= 256 for token in record.output_token_ids)
        kinds = [item.kind for item in record.lineage]
        assert kinds == ["project-fixture", "checkpoint"]
        assert record.created_at is not None and record.prompt_template_sha256
    # The same seed reproduces the same tokens (records differ only by time).
    second = sample_teacher(fixture.store, teacher, _prompts(), sampling)
    assert [fixture.store.get(i).output_token_ids for i in second] == [
        r.output_token_ids for r in again
    ]


def _register_source(fixture: AlignmentFixture, source_id: str, uses: list[str]) -> None:
    registration = tokenizer_registration(source_id, "teacher-family").model_copy(
        update={"media_type": "application/jsonl", "allowed_uses": uses}
    )
    fixture.store.data.register(registration)


def _teacher_yaml(path: Path, source_id: str, **overrides: Any) -> Path:
    values: dict[str, Any] = {
        "schema_version": 1,
        "id": f"teacher-{source_id}",
        "source_id": source_id,
        "provider": "Example Lab",
        "model": "example-model",
        "model_version": "2026-01",
        "rights": {
            "license_id": "example-outputs-licence",
            "training_on_outputs_permitted": True,
            "permission_basis": "Written permission to train on these outputs (test).",
        },
        "prompt_template_id": "operator-template-1",
        "prompt_template_sha256": "d" * 64,
        "sampling": {"decoding": "external"},
        **overrides,
    }
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return path


def test_registered_teacher_outputs_need_reviewed_training_rights(
    fixture: AlignmentFixture, tmp_path: Path
) -> None:
    with pytest.raises(TeacherRefusedError, match="no data-engine registration"):
        register_teacher(fixture.store, _teacher_yaml(tmp_path / "a.yaml", "unregistered"))
    _register_source(fixture, "eval-only-outputs", ["research", "evaluate"])
    with pytest.raises(TeacherRefusedError, match="not registered for training"):
        register_teacher(fixture.store, _teacher_yaml(tmp_path / "b.yaml", "eval-only-outputs"))
    _register_source(fixture, "no-output-rights", ["research", "train"])
    with pytest.raises(ValueError, match="does not permit training on its outputs"):
        register_teacher(
            fixture.store,
            _teacher_yaml(
                tmp_path / "c.yaml",
                "no-output-rights",
                rights={
                    "license_id": "x",
                    "training_on_outputs_permitted": False,
                    "permission_basis": "terms forbid it",
                },
            ),
        )
    with pytest.raises(ValueError, match="cost-approval"):
        register_teacher(
            fixture.store,
            _teacher_yaml(
                tmp_path / "d.yaml",
                "no-output-rights",
                rights={
                    "license_id": "x",
                    "training_on_outputs_permitted": True,
                    "permission_basis": "paid access",
                    "cost": "paid",
                },
            ),
        )


def test_imported_teacher_outputs_carry_lineage_and_tombstones_propagate(
    fixture: AlignmentFixture, tmp_path: Path
) -> None:
    _register_source(fixture, "licensed-outputs", ["research", "train"])
    record_id, teacher = register_teacher(
        fixture.store, _teacher_yaml(tmp_path / "t.yaml", "licensed-outputs")
    )
    assert teacher.rights.registration_sha256 is not None
    with pytest.raises(TeacherRefusedError, match="immutable"):
        register_teacher(fixture.store, _teacher_yaml(tmp_path / "t.yaml", "licensed-outputs"))
    outputs = tmp_path / "outputs.jsonl"
    stamp = datetime(2026, 10, 1, tzinfo=UTC).isoformat()
    lines = [
        {
            "messages": [{"role": "user", "content": f"Name a colour, number {n}."}],
            "output": f"Blue, number {n}.",
            "sampled_at": stamp,
        }
        for n in range(3)
    ]
    outputs.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")
    imported = import_teacher_outputs(fixture.store, teacher.id, outputs)
    record = fixture.store.get(imported[0])
    assert record.generator.kind == "registered-output-file"
    assert (record.generator.provider, record.generator.model) == ("Example Lab", "example-model")
    assert {item.kind for item in record.lineage} == {"source_id", "raw_sha256", "alignment_record"}
    assert record_id in {item.id for item in record.lineage}

    derived = fixture.store.put(
        PreferenceRecord(
            origin="human",
            messages=record.messages,
            chosen=record.output,
            rejected="Not a colour.",
            lineage=(parent(imported[0]),),
        )
    )
    index = fixture.store.removal_index()
    assert not index.removed(imported[0]) and not index.removed(derived)
    DataEngine(fixture.store.data.root).remove(
        target_kind="source_id",
        target="licensed-outputs",
        reason="licence withdrawn (test)",
        requested_by="test",
    )
    index = fixture.store.removal_index()
    assert index.removed(record_id)
    assert all(index.removed(item) for item in imported)
    assert index.removed(derived)
    with pytest.raises(TeacherRefusedError, match="removed"):
        import_teacher_outputs(fixture.store, teacher.id, outputs)


# --- distillation ---------------------------------------------------------------


def test_distillation_kl_is_zero_for_identical_models_and_ignores_masked_targets() -> None:
    logits = torch.randn(2, 5, 11)
    targets = torch.randint(0, 11, (2, 5))
    assert float(distillation_kl_sum(logits, [logits], targets, 2.0)) == pytest.approx(0, abs=1e-5)
    masked = torch.full_like(targets, -100)
    assert float(distillation_kl_sum(logits, [torch.randn(2, 5, 11)], masked, 1.0)) == 0.0
    other = torch.randn(2, 5, 11)
    manual = torch.nn.functional.kl_div(
        torch.log_softmax(logits.reshape(-1, 11), -1),
        torch.softmax(other.reshape(-1, 11), -1),
        reduction="sum",
    )
    assert float(distillation_kl_sum(logits, [other], targets, 1.0)) == pytest.approx(
        float(manual), rel=1e-5
    )


def test_response_distillation_from_two_teachers_trains_a_fresh_student(
    fixture: AlignmentFixture, sft_checkpoint: Path, tmp_path: Path
) -> None:
    sampling = Sampling(decoding="greedy", max_new_tokens=10)
    first = sample_teacher(
        fixture.store,
        load_chat_model(fixture.base_checkpoint, fixture.shards.data_root),
        _prompts(4),
        sampling,
    )
    second = sample_teacher(
        fixture.store,
        load_chat_model(sft_checkpoint, fixture.shards.data_root),
        _prompts(4),
        sampling,
    )
    dataset_id, manifest = build_sft_dataset(
        fixture.store, fixture.tokenizer, [*first, *second], validation_percent=0
    )
    generators = {
        fixture.store.get(i).generator.checkpoint_sha256
        for i in manifest.splits["train"].record_ids
    }
    assert len(generators) == 2  # every example keeps its own teacher record
    student = kernel_config(fixture.shards.vocab_size, d_model=32, d_ff=96)
    result = pretrain_packed(
        student,
        fixture.shards.data_root,
        tmp_path / "student",
        sft_settings(fixture, dataset_id, init_checkpoint=None, validation_split=None),
        max_steps=2,
    )
    assert len(result.losses) == 2


def test_logit_distillation_adds_a_kl_term_from_an_epor_teacher(
    fixture: AlignmentFixture, tmp_path: Path
) -> None:
    settings = kernel_settings(
        fixture.shards.tokenized_id,
        distillation=LogitDistillation(teachers=(fixture.base_ref,), weight=0.5, temperature=2.0),
    )
    result = pretrain_packed(
        fixture.config, fixture.shards.data_root, tmp_path / "distil", settings, max_steps=2
    )
    metrics = [
        json.loads(line)
        for line in (tmp_path / "distil" / "metrics.jsonl").read_text().splitlines()
    ]
    assert all(item["distill_kl"] > 0 for item in metrics)
    assert result.losses[0] == pytest.approx(
        metrics[0]["main_loss"] + 0.5 * metrics[0]["distill_kl"]
    )
    # Resume keeps the teachers in the trajectory: exact continuation.
    pretrain_packed(
        fixture.config, fixture.shards.data_root, tmp_path / "half", settings, max_steps=1
    )
    resumed = pretrain_packed(
        fixture.config,
        fixture.shards.data_root,
        tmp_path / "half",
        settings,
        max_steps=2,
        resume_from=tmp_path / "half",
    )
    assert resumed.losses == result.losses[1:]


def test_logit_distillation_is_refused_across_tokenizers(
    fixture: AlignmentFixture, tmp_path: Path
) -> None:
    from epor.tokenizer import build_token_shards, load_tokenized_manifest, train_tokenizer

    data = fixture.shards.data_root
    manifest = load_tokenized_manifest(data, fixture.shards.tokenized_id)
    other = train_tokenizer(data, manifest.dataset_id, manifest.build_id, vocab_size=800)
    other_shards = build_token_shards(
        data, manifest.dataset_id, manifest.build_id, other.tokenizer_id
    )
    teacher_run = tmp_path / "other-teacher"
    pretrain_packed(
        kernel_config(other_shards.manifest.vocab_size),
        data,
        teacher_run,
        kernel_settings(other_shards.tokenized_id),
        max_steps=1,
    )
    directory = latest_checkpoint(teacher_run)
    assert directory is not None
    settings = kernel_settings(
        fixture.shards.tokenized_id,
        distillation=LogitDistillation(teachers=(checkpoint_ref(directory),)),
    )
    with pytest.raises(DistillationRefusedError, match="share the EPOR tokenizer"):
        pretrain_packed(fixture.config, data, tmp_path / "refused", settings, max_steps=1)


# --- DPO ------------------------------------------------------------------------


def _dpo(sft: Path, dataset: str, **overrides: Any) -> DpoSettings:
    ref = checkpoint_ref(sft)
    values: dict[str, Any] = {
        "dataset_id": dataset,
        "policy_init": ref,
        "reference": ref,
        "pairs_per_step": 3,
        "optimizer": OptimizerSettings(learning_rate=1e-3, weight_decay=0.0, total_steps=6),
        "checkpoint_every": 2,
        **overrides,
    }
    return DpoSettings.model_validate(values)


def test_dpo_terms_start_at_log_two_and_reward_the_chosen_reply() -> None:
    zeros = torch.zeros(3, 2)
    terms = dpo_terms(zeros, zeros, torch.ones(3, 2), 0.1)
    assert float(terms["loss"]) == pytest.approx(0.693147, abs=1e-6)
    better = torch.tensor([[1.0, -1.0]] * 3)
    assert float(dpo_terms(better, zeros, torch.ones(3, 2), 0.1)["accuracy"]) == 1.0


def test_dpo_trains_from_sft_against_a_frozen_reference_and_resumes_exactly(
    fixture: AlignmentFixture, sft_checkpoint: Path, preference_dataset: str, tmp_path: Path
) -> None:
    settings = _dpo(sft_checkpoint, preference_dataset)
    data = fixture.shards.data_root
    full = train_preference(data, tmp_path / "full", settings, max_steps=4)
    assert full.losses[0] == pytest.approx(0.693147, abs=1e-5)  # policy == reference at step 1
    assert full.evaluation is not None and full.evaluation.pairs > 0
    manifest = json.loads((tmp_path / "full" / "manifest.json").read_text())
    assert manifest["command"] == "alignment_dpo" and manifest["status"] == "succeeded"
    assert "not evidence of alignment" in manifest["claim"]

    train_preference(data, tmp_path / "split", settings, max_steps=2)
    resumed = train_preference(
        data, tmp_path / "split", settings, max_steps=4, resume_from=tmp_path / "split"
    )
    assert resumed.losses == full.losses[2:]
    assert same_weights(final_model_state(tmp_path / "split"), final_model_state(tmp_path / "full"))
    # The reference is read, never written.
    reference_after = load_chat_model(sft_checkpoint, data)
    assert reference_after.checkpoint_sha256 == checkpoint_ref(sft_checkpoint).checkpoint_sha256
    # The result is an ordinary kernel checkpoint the chat loader can read.
    load_chat_model(tmp_path / "full", data)
    with pytest.raises(CheckpointError, match="preference"):
        resume_inputs(tmp_path / "full")
    with pytest.raises(CheckpointError, match="trajectory"):
        train_preference(
            data,
            tmp_path / "split",
            settings.model_copy(update={"beta": 0.5}),
            max_steps=6,
            resume_from=tmp_path / "split",
        )


def test_dpo_monitor_abort_stops_at_a_checkpoint(
    fixture: AlignmentFixture, sft_checkpoint: Path, preference_dataset: str, tmp_path: Path
) -> None:
    settings = _dpo(
        sft_checkpoint,
        preference_dataset,
        optimizer=OptimizerSettings(learning_rate=5e-2, weight_decay=0.0, total_steps=6),
        monitor=DpoMonitor(logratio_alert=1e-9, logratio_abort=1e-6),
    )
    with pytest.raises(MonitorAbortError, match="abort threshold"):
        train_preference(fixture.shards.data_root, tmp_path / "abort", settings, max_steps=6)
    manifest = json.loads((tmp_path / "abort" / "manifest.json").read_text())
    assert manifest["status"] == "aborted" and manifest["termination_reason"] == "monitor-abort"
    assert manifest["alerts"]
    assert latest_checkpoint(tmp_path / "abort") is not None


def test_dpo_cancellation_checkpoints_the_last_step(
    fixture: AlignmentFixture, sft_checkpoint: Path, preference_dataset: str, tmp_path: Path
) -> None:
    calls = {"n": 0}

    def cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(TrainingCancelled):
        train_preference(
            fixture.shards.data_root,
            tmp_path / "cancel",
            _dpo(sft_checkpoint, preference_dataset),
            max_steps=6,
            should_cancel=cancel,
        )
    directory = latest_checkpoint(tmp_path / "cancel")
    assert directory is not None and directory.name == "step-00000001"


def test_dpo_refuses_an_sft_dataset_and_a_mismatched_reference(
    fixture: AlignmentFixture,
    sft_checkpoint: Path,
    seeded: dict[str, list[str]],
    tmp_path: Path,
) -> None:
    from epor.training.chat_data import ChatDatasetError

    sft_dataset, _ = build_sft_dataset(fixture.store, fixture.tokenizer, seeded["examples"])
    with pytest.raises(ChatDatasetError, match="sft dataset"):
        train_preference(
            fixture.shards.data_root, tmp_path / "x", _dpo(sft_checkpoint, sft_dataset), max_steps=1
        )
    wrong = checkpoint_ref(sft_checkpoint).model_copy(update={"checkpoint_sha256": "e" * 64})
    with pytest.raises(CheckpointError, match="pinned"):
        train_preference(
            fixture.shards.data_root,
            tmp_path / "y",
            _dpo(sft_checkpoint, sft_dataset, reference=wrong),
            max_steps=1,
        )


# --- SFT across topologies --------------------------------------------------------


def test_sft_agrees_exactly_across_single_process_ddp_and_fsdp2(
    fixture: AlignmentFixture, seeded: dict[str, list[str]], tmp_path: Path
) -> None:
    from kernel_support import train_worker

    from epor.training import launch_local

    dataset_id, _ = build_sft_dataset(fixture.store, fixture.tokenizer, seeded["examples"])
    settings = sft_settings(fixture, dataset_id, validation_split=None)
    single = pretrain_packed(
        fixture.config, fixture.shards.data_root, tmp_path / "single", settings, max_steps=3
    )
    jobs = [
        {
            "strategy": strategy,
            "config": fixture.config,
            "data_root": fixture.shards.data_root,
            "output": tmp_path / strategy,
            "settings": settings,
            "max_steps": 3,
        }
        for strategy in ("ddp", "fsdp2")
    ]
    ranks = launch_local(train_worker, 2, jobs, timeout_seconds=180, threads_per_rank=2)
    assert ranks[0] == ranks[1]
    assert ranks[0][0] == single.losses and ranks[0][1] == single.losses
    reference = final_model_state(tmp_path / "single")
    assert same_weights(final_model_state(tmp_path / "ddp"), reference)
    assert same_weights(final_model_state(tmp_path / "fsdp2"), reference)


def test_a_fully_masked_micro_batch_contributes_an_exact_zero(fixture: AlignmentFixture) -> None:
    from epor.models.reference import EporDecoder
    from epor.training.kernel import micro_objective

    model = EporDecoder(fixture.config, seed=1)
    inputs = torch.randint(256, 300, (1, 8))
    output = model(inputs, torch.full((1, 8), -100))
    objective = micro_objective(
        output,
        target_tokens=0,
        mtp_counts=[],
        global_target_tokens=10,
        global_mtp_counts=[],
        mtp_loss_weight=0.0,
    )
    objective.backward()
    assert float(objective) == 0.0
    assert all(p.grad is not None and float(p.grad.abs().sum()) == 0.0 for p in model.parameters())
