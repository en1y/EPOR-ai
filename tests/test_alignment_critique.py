"""Constitutional critique/revision, AI judging, human audits, and overoptimization monitors."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

torch = pytest.importorskip("torch")

from alignment_support import AlignmentFixture, make_alignment_fixture  # noqa: E402
from kernel_support import PROJECT_ROOT  # noqa: E402

from epor.alignment.audit import (  # noqa: E402
    AuditError,
    AuditLedger,
    admission_gate,
    calibration,
    pending,
    sample_run,
)
from epor.alignment.critique import covenant_obligations, critique_and_revise  # noqa: E402
from epor.alignment.datasets import build_preference_dataset, build_sft_dataset  # noqa: E402
from epor.alignment.fixtures import load_fixtures, seed_fixtures  # noqa: E402
from epor.alignment.judge import (  # noqa: E402
    G10_CITATIONS,
    generate_eval_items,
    judge_eval_items,
    judge_pairs,
)
from epor.alignment.model import load_chat_model  # noqa: E402
from epor.alignment.monitors import (  # noqa: E402
    Threshold,
    audited_pairs,
    classify,
    held_out_pairs,
    load_monitor_config,
    load_report,
    overoptimization_report,
)
from epor.alignment.preference import DpoMonitor  # noqa: E402
from epor.alignment.records import (  # noqa: E402
    AlignmentRecordError,
    JudgementRecord,
    PreferenceRecord,
    Sampling,
    parent,
)
from epor.control.authority import ANONYMOUS, LOCAL_CLI  # noqa: E402
from epor.safety import load_covenant  # noqa: E402

GREEDY = Sampling(decoding="greedy", max_new_tokens=8)


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> AlignmentFixture:
    return make_alignment_fixture(tmp_path_factory.mktemp("alignment-critique"))


@pytest.fixture(scope="module")
def seeded(fixture: AlignmentFixture) -> dict[str, list[str]]:
    return seed_fixtures(fixture.store)


@pytest.fixture(scope="module")
def model(fixture: AlignmentFixture) -> Any:
    return load_chat_model(fixture.base_checkpoint, fixture.shards.data_root)


# --- the covenant is cited, never copied --------------------------------------------


def _windows(text: str, size: int = 6) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z']+", text.lower())
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


def test_alignment_code_reads_the_covenant_and_holds_no_copy_of_its_text() -> None:
    covenant = yaml.safe_load((PROJECT_ROOT / "configs" / "covenant-v1.yaml").read_text())
    protected: set[tuple[str, ...]] = set()
    for principle in covenant["principles"]:
        protected |= _windows(principle["law"])
        for obligation in principle["obligations"]:
            protected |= _windows(obligation["statement"])
    assert protected
    sources = sorted((PROJECT_ROOT / "src" / "epor" / "alignment").glob("*.py")) + sorted(
        (PROJECT_ROOT / "configs" / "alignment").glob("*.yaml")
    )
    assert len(sources) > 5
    for path in sources:
        shared = _windows(path.read_text(encoding="utf-8")) & protected
        assert not shared, f"{path.name} restates covenant text: {sorted(shared)[:2]}"
    # The text it uses is the covenant file's own, by key.
    loaded = load_covenant()
    obligations = covenant_obligations()
    for principle in loaded.principles:
        for obligation in principle.obligations:
            assert obligations[obligation.key] == (principle.priority, obligation.statement)
    assert {key for _, key in G10_CITATIONS.values()} <= set(obligations)


def test_critique_and_revision_are_covenant_citing_candidates(
    fixture: AlignmentFixture, seeded: dict[str, list[str]], model: Any
) -> None:
    covenant = load_covenant()
    written = critique_and_revise(
        fixture.store, model, seeded["examples"][:2], ["no-harmful-output"], GREEDY
    )
    assert len(written) == 2
    for entry in written:
        critique = fixture.store.get(entry["critique"])
        revision = fixture.store.get(entry["revision"])
        for record in (critique, revision):
            assert record.candidate
            assert record.covenant.obligations == ("no-harmful-output",)
            assert record.covenant.covenant_sha256 == covenant.sha256
            assert "covenant" in {item.kind for item in record.lineage}
        assert revision.messages == fixture.store.get(entry["example"]).messages
        assert entry["critique"] in {item.id for item in revision.lineage}
    with pytest.raises(ValueError, match="not covenant obligation keys"):
        critique_and_revise(fixture.store, model, seeded["examples"][:1], ["be-nice"], GREEDY)
    with pytest.raises(AlignmentRecordError, match="never training targets"):
        build_sft_dataset(fixture.store, fixture.tokenizer, [written[0]["critique"]])
    # Without an audited judging run a revision never reaches a dataset.
    with pytest.raises(ValueError, match="no admissible record"):
        build_sft_dataset(fixture.store, fixture.tokenizer, [written[0]["revision"]])


def test_the_judge_records_decisions_as_data_and_writes_ai_judged_pairs(
    fixture: AlignmentFixture, seeded: dict[str, list[str]], model: Any
) -> None:
    written = critique_and_revise(
        fixture.store, model, seeded["examples"][2:5], ["no-harm-through-inaction"], GREEDY
    )
    pairs = [(entry["example"], entry["revision"]) for entry in written]
    result = judge_pairs(
        fixture.store,
        model,
        pairs,
        obligation="no-harm-through-inaction",
        min_margin=0.0,
        subject="revision",
    )
    assert len(result["judgements"]) == 3
    for judgement_id in result["judgements"]:
        judgement = fixture.store.get(judgement_id)
        assert judgement.judging_run == result["judging_run"]
        assert judgement.generator.checkpoint_sha256 == model.checkpoint_sha256
        assert 0.5 <= judgement.confidence <= 1.0
        assert judgement.accepted == (judgement.preferred == judgement.inputs[1])
    for preference_id in result["preferences"]:
        preference = fixture.store.get(preference_id)
        assert preference.origin == "ai-judge" and preference.judgement in result["judgements"]
    with pytest.raises(ValueError, match="not a covenant obligation"):
        judge_pairs(fixture.store, model, pairs, obligation="unknown-key")


# --- audits -------------------------------------------------------------------------


def _synthetic_run(
    fixture: AlignmentFixture, seeded: dict[str, list[str]], decisions: list[bool], tag: str
) -> tuple[str, list[str], list[str]]:
    """A judging run with fixed accept/reject decisions over fixture-derived pairs."""

    run = hashlib.sha256(tag.encode()).hexdigest()
    judgements, preferences = [], []
    for index, accepted in enumerate(decisions):
        base = fixture.store.get(seeded["examples"][index])
        other = fixture.store.put(
            base.model_copy(update={"output": f"{base.output} ({tag} variant {index})"})
        )
        judgement = fixture.store.put(
            JudgementRecord(
                judging_run=run,
                subject="preference",
                generator=base.generator,
                prompt_template_id="t",
                prompt_template_sha256="1" * 64,
                inputs=(seeded["examples"][index], other),
                accepted=accepted,
                preferred=seeded["examples"][index],
                margin=0.3 if accepted else 0.01,
                confidence=0.9 if accepted else 0.55,
            )
        )
        judgements.append(judgement)
        if accepted:
            preferences.append(
                fixture.store.put(
                    PreferenceRecord(
                        origin="ai-judge",
                        messages=base.messages,
                        chosen=base.output,
                        rejected=f"{base.output} ({tag} variant {index})",
                        judgement=judgement,
                        lineage=(parent(seeded["examples"][index]), parent(other)),
                    )
                )
            )
    return run, judgements, preferences


def test_sampling_queues_both_strata_and_nothing_is_ever_auto_audited(
    fixture: AlignmentFixture, seeded: dict[str, list[str]], tmp_path: Path
) -> None:
    ledger = AuditLedger(tmp_path / "safety")
    run, _, preferences = _synthetic_run(fixture, seeded, [True, True, True, False, False], "abc")
    queued = sample_run(ledger, fixture.store, run, per_stratum=2, seed=3, actor_id="owner")
    assert len(queued) == 4
    assert sample_run(ledger, fixture.store, run, per_stratum=2, seed=3, actor_id="o") == []
    items = ledger.items()
    assert {item.stratum for item in items} == {"accepted", "rejected"}
    assert all(item.verdict is None for item in items) and len(pending(ledger)) == 4
    assert all("replies" in item.detail for item in items)
    gate = admission_gate(ledger, fixture.store, max_disagreement=0.25)
    decision = gate(preferences[0], fixture.store.get(preferences[0]))
    assert not decision.admitted and "await a human verdict" in decision.reason
    _, manifest = build_preference_dataset(
        fixture.store, fixture.tokenizer, [*seeded["preferences"], *preferences], gate=gate
    )
    assert manifest.excluded == {"candidate-not-admitted": len(preferences)}


def test_verdicts_need_a_verified_human_and_drive_calibration_and_admission(
    fixture: AlignmentFixture, seeded: dict[str, list[str]], tmp_path: Path
) -> None:
    ledger = AuditLedger(tmp_path / "safety")
    run, _, preferences = _synthetic_run(fixture, seeded, [True, True, False, False], "def")
    sample_run(ledger, fixture.store, run, per_stratum=2, seed=0, actor_id="owner")
    items = {item.record: item for item in ledger.items()}
    with pytest.raises(AuditError, match="verified human"):
        ledger.record_verdict(next(iter(items.values())).id, "accept", actor=ANONYMOUS)
    with pytest.raises(AuditError, match="not found"):
        ledger.record_verdict("missing", "accept", actor=LOCAL_CLI)
    for item in items.values():
        ledger.record_verdict(
            item.id, "accept" if item.judge_accepted else "reject", actor=LOCAL_CLI
        )
    report = calibration(ledger, run, store=fixture.store)
    assert report["audited"] == 4 and report["pending"] == 0 and report["decisions"] == 4
    assert report["disagreement_rate"] == 0.0
    assert report["agreement_matrix"]["judge_accept__human_accept"] == 2
    gate = admission_gate(ledger, fixture.store, max_disagreement=0.25)
    _, manifest = build_preference_dataset(
        fixture.store, fixture.tokenizer, [*seeded["preferences"], *preferences], gate=gate
    )
    assert manifest.excluded == {}
    assert manifest.admission["candidates"][0]["disagreement_rate"] == 0.0

    # A later human rejection of one candidate's own decision excludes it and
    # raises the disagreement rate; the earlier verdict stays on record.
    first = items[fixture.store.get(preferences[0]).judgement]
    ledger.record_verdict(first.id, "reject", actor=LOCAL_CLI, note="on reflection")
    assert len(ledger.item(first.id).verdicts) == 2
    assert calibration(ledger, run)["disagreement_rate"] == 0.25
    gate = admission_gate(ledger, fixture.store, max_disagreement=0.25)
    assert not gate(preferences[0], fixture.store.get(preferences[0])).admitted
    assert gate(preferences[1], fixture.store.get(preferences[1])).admitted
    strict = admission_gate(ledger, fixture.store, max_disagreement=0.1)
    assert "exceeds" in strict(preferences[1], fixture.store.get(preferences[1])).reason


def test_a_run_without_rejected_decisions_cannot_be_admitted(
    fixture: AlignmentFixture, seeded: dict[str, list[str]], tmp_path: Path
) -> None:
    ledger = AuditLedger(tmp_path / "safety")
    run, _, preferences = _synthetic_run(fixture, seeded, [True, True], "fed")
    sample_run(ledger, fixture.store, run, per_stratum=2, seed=0, actor_id="owner")
    for item in ledger.items():
        ledger.record_verdict(item.id, "accept", actor=LOCAL_CLI)
    decision = admission_gate(ledger, fixture.store, max_disagreement=1.0)(
        preferences[0], fixture.store.get(preferences[0])
    )
    assert not decision.admitted and "rejected decision(s) audited" in decision.reason


def test_model_written_eval_items_are_candidates_judged_and_audited(
    fixture: AlignmentFixture, model: Any, tmp_path: Path
) -> None:
    items = generate_eval_items(fixture.store, model, 6, GREEDY)
    records = [fixture.store.get(item) for item in items]
    assert {record.category for record in records} == set(G10_CITATIONS)
    assert all(record.generator.kind == "epor-checkpoint" for record in records)
    result = judge_eval_items(fixture.store, model, items, min_margin=0.0)
    ledger = AuditLedger(tmp_path / "safety")
    sample_run(
        ledger, fixture.store, result["judging_run"], per_stratum=3, seed=1, actor_id="owner"
    )
    queued = ledger.items()
    assert queued and all(item.subject == "eval-item" and item.verdict is None for item in queued)
    assert "question" in queued[0].detail


# --- monitors -----------------------------------------------------------------------


def test_monitor_thresholds_are_hypotheses_and_match_the_dpo_defaults() -> None:
    config = load_monitor_config()
    assert config.evidence == "hypothesis"
    defaults = DpoMonitor()
    assert (defaults.logratio_alert, defaults.logratio_abort) == (
        config.dpo_step_logratio.alert,
        config.dpo_step_logratio.abort,
    )
    threshold = Threshold(alert=1.0, abort=2.0)
    assert [classify(v, threshold) for v in (None, 0.5, 1.0, 2.5)] == [
        "unknown",
        "ok",
        "alert",
        "abort",
    ]


def test_overoptimization_report_is_stored_and_flags_unknown_without_audits(
    fixture: AlignmentFixture, seeded: dict[str, list[str]], model: Any, tmp_path: Path
) -> None:
    dataset_id, _ = build_preference_dataset(
        fixture.store, fixture.tokenizer, seeded["preferences"], validation_percent=35
    )
    pairs = held_out_pairs(fixture.store, dataset_id)
    assert pairs
    prompts = [item.messages for item in load_fixtures().prompts[:3]]
    report_id, report = overoptimization_report(
        fixture.store,
        model,
        model,
        prompts=prompts,
        held_out_pairs=pairs,
        audited_pairs=audited_pairs(fixture.store, AuditLedger(tmp_path / "safety")),
        beta=0.1,
        max_new_tokens=6,
    )
    monitors = report["monitors"]
    assert monitors["kl_per_token"]["value"] == pytest.approx(0.0, abs=1e-6)
    assert monitors["kl_per_token"]["status"] == "ok"
    assert monitors["proxy_minus_audited"]["status"] == "unknown"
    assert monitors["proxy_minus_audited"]["audited_agreement"] is None
    assert report["status"] == "unknown"
    assert report["thresholds"]["evidence"] == "hypothesis"
    assert load_report(fixture.store, "overoptimization", report_id) == report
    path = fixture.store.data.alignment_path(f"reports/overoptimization/{report_id}.json")
    path.chmod(0o600)
    path.write_text(path.read_text().replace('"ok"', '"abort"'))
    with pytest.raises(ValueError, match="altered"):
        load_report(fixture.store, "overoptimization", report_id)


def test_only_accepted_verdicts_endorse_an_audited_pair(
    fixture: AlignmentFixture, seeded: dict[str, list[str]], tmp_path: Path
) -> None:
    ledger = AuditLedger(tmp_path / "safety")
    run, judgements, _ = _synthetic_run(fixture, seeded, [True, False], "cab")
    sample_run(ledger, fixture.store, run, per_stratum=1, seed=0, actor_id="owner")
    items = {item.record: item for item in ledger.items()}
    ledger.record_verdict(items[judgements[0]].id, "accept", actor=LOCAL_CLI)
    # A rejection says the judge's pick is not better; it endorses nothing.
    ledger.record_verdict(items[judgements[1]].id, "reject", actor=LOCAL_CLI)
    [(_, endorsed, other)] = audited_pairs(fixture.store, ledger)
    judgement = fixture.store.get(judgements[0])
    assert endorsed == fixture.store.get(judgement.preferred).output
    assert other != endorsed
