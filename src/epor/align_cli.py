"""Covenant-gated knowledge-transfer and alignment commands (v0.0.7).

``epor train sft|preference`` and ``epor align ...``.  Every command resolves
its declared action (:data:`epor.actions.CLI_ACTIONS`) before doing anything,
and every operator-supplied path must resolve beneath ``EPOR_PROJECT_ROOT``.
Human audit verdicts recorded here are the owner's own (the local CLI actor).

Exit codes: 0 success, 1 any other failure (including a G10 comparison that
fails closed and an overoptimization report at ``abort``), 2 input or
verification failure, 3 refusal by the covenant or a rights control, 4 a G10
comparison escalated to human review, 130 stopped by SIGINT/SIGTERM.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Annotated, Any, Literal

import typer

align_app = typer.Typer(
    help="Knowledge transfer and alignment: teachers, critique, judging, audits, monitors, G10."
)
teacher_app = typer.Typer(help="Register, import, and sample teachers.")
source_app = typer.Typer(help="Import operator-registered training sources.")
dataset_app = typer.Typer(help="Build SFT and preference datasets from provenance records.")
audit_app = typer.Typer(help="Human audits of AI judge decisions.")
align_app.add_typer(teacher_app, name="teacher")
align_app.add_typer(source_app, name="source")
align_app.add_typer(dataset_app, name="dataset")
align_app.add_typer(audit_app, name="audit")

_ESCALATED_EXIT = 4
_CANCELLED_EXIT = 130


def _cli() -> Any:
    import epor.cli

    return epor.cli


def _refuse(message: str, *, code: int = 2) -> typer.Exit:
    _cli().console.print(f"[red]refused[/red]: {message}")
    return typer.Exit(code=code)


def _failure(exc: BaseException) -> typer.Exit:
    from epor.alignment.audit import AuditError
    from epor.alignment.records import AlignmentRecordError
    from epor.alignment.teachers import TeacherRefusedError
    from epor.data.store import DataStoreError
    from epor.training.kernel import DistillationRefusedError
    from epor.training.runner import is_cancellation

    if is_cancellation(exc) or isinstance(exc, InterruptedError):
        typer.echo(f"stopped: {exc}", err=True)
        return typer.Exit(code=_CANCELLED_EXIT)
    if isinstance(exc, (TeacherRefusedError, DistillationRefusedError)):
        return _refuse(str(exc), code=3)
    if isinstance(
        exc,
        (
            AlignmentRecordError,
            AuditError,
            DataStoreError,
            FileExistsError,
            FileNotFoundError,
            NotADirectoryError,
            ValueError,
        ),
    ):
        return _refuse(str(exc))
    _cli().console.print(f"[red]failed[/red] ({type(exc).__name__}): {exc}")
    return typer.Exit(code=1)


def _run(work: Any) -> Any:
    """Run ``work(should_cancel)`` with SIGINT/SIGTERM mapped to a cooperative stop."""

    stop = threading.Event()
    try:
        with _cli()._StopRequests(stop.set):
            return work(stop.is_set)
    except typer.Exit:
        raise
    except Exception as exc:
        raise _failure(exc) from exc


def _store() -> Any:
    from epor.alignment.records import AlignmentStore

    return AlignmentStore(_cli()._data_root())


def _ledger() -> Any:
    from epor.alignment.audit import AuditLedger
    from epor.control.settings import ControlSettings

    settings = ControlSettings.from_environment()
    settings.prepare_directories()
    assert settings.safety_root is not None
    return AuditLedger(settings.safety_root)


def _checkpoint(path: Path) -> Path:
    from epor.training.run_checkpoint import resolve_checkpoint

    directory = _cli()._contained_directory(path, label="checkpoint path")
    try:
        return resolve_checkpoint(directory)
    except Exception as exc:
        raise _failure(exc) from exc


def _model(path: Path) -> Any:
    from epor.alignment.model import load_chat_model

    checkpoint = _checkpoint(path)
    try:
        return load_chat_model(checkpoint, _cli()._data_root())
    except Exception as exc:
        raise _failure(exc) from exc


def _ref(path: Path) -> Any:
    from epor.alignment.model import checkpoint_identity
    from epor.training.kernel import CheckpointRef

    directory = _checkpoint(path)
    return CheckpointRef(path=str(directory), checkpoint_sha256=checkpoint_identity(directory))


def _sampling(temperature: float, top_k: int | None, max_new_tokens: int, seed: int) -> Any:
    from epor.alignment.records import Sampling

    if temperature == 0:
        return Sampling(decoding="greedy", max_new_tokens=max_new_tokens, seed=seed)
    return Sampling(
        decoding="temperature",
        temperature=temperature,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )


def _echo(payload: object) -> None:
    _cli()._echo_json(payload)


_Temperature = Annotated[float, typer.Option(min=0.0, max=10.0, help="0 decodes greedily.")]
_TopK = Annotated[int | None, typer.Option(min=1, max=65_536)]
_MaxNew = Annotated[int, typer.Option(min=1, max=512)]
_Seed = Annotated[int, typer.Option(min=0, max=2**32 - 1)]


# --------------------------------------------------------------------- records


@align_app.command("fixtures")
def align_fixtures() -> None:
    """Write the project-authored demonstrations and preference pairs as records."""

    _cli().authorize_command("align fixtures")
    from epor.alignment.fixtures import seed_fixtures

    try:
        written = seed_fixtures(_store())
    except Exception as exc:
        raise _failure(exc) from exc
    _echo({"examples": written["examples"], "preferences": written["preferences"]})


@teacher_app.command("register")
def teacher_register(
    registration: Annotated[Path, typer.Argument(dir_okay=False, help="Teacher YAML/JSON.")],
) -> None:
    """Register a rights-reviewed teacher-output source (CLI only)."""

    cli = _cli()
    cli.authorize_command("align teacher register")
    from epor.alignment.teachers import register_teacher

    path = cli._reference_path(registration, label="teacher registration", must_exist=True)[1]
    try:
        record_id, teacher = register_teacher(_store(), path)
    except Exception as exc:
        raise _failure(exc) from exc
    _echo({"record": record_id, "teacher": teacher.id, "source_id": teacher.source_id})


@teacher_app.command("import")
def teacher_import(
    teacher_id: Annotated[str, typer.Argument()],
    outputs: Annotated[Path, typer.Argument(dir_okay=False, help="JSONL teacher outputs.")],
) -> None:
    """Import a registered teacher's outputs as provenance records."""

    cli = _cli()
    cli.authorize_command("align teacher import")
    from epor.alignment.teachers import import_teacher_outputs

    path = cli._reference_path(outputs, label="teacher outputs", must_exist=True)[1]
    try:
        written = import_teacher_outputs(_store(), teacher_id, path)
    except Exception as exc:
        raise _failure(exc) from exc
    _echo({"teacher": teacher_id, "examples": written})


@source_app.command("import")
def source_import(
    source_id: Annotated[str, typer.Argument()],
    path: Annotated[Path, typer.Argument(dir_okay=False)],
    fmt: Annotated[
        Literal["epor-chat-jsonl", "epor-preference-jsonl"], typer.Option("--format")
    ] = "epor-chat-jsonl",
) -> None:
    """Import demonstrations or preference pairs from a source registered for training."""

    cli = _cli()
    cli.authorize_command("align source import")
    from epor.alignment.teachers import import_registered_source

    resolved = cli._reference_path(path, label="source file", must_exist=True)[1]
    try:
        written = import_registered_source(_store(), source_id, resolved, fmt)
    except Exception as exc:
        raise _failure(exc) from exc
    _echo({"source_id": source_id, "records": written})


@teacher_app.command("sample")
def teacher_sample(
    checkpoint: Annotated[Path, typer.Argument(help="Teacher run or checkpoint directory.")],
    prompt_record: Annotated[
        list[str] | None,
        typer.Option("--prompt-record", help="Reuse an example record's prompt; repeatable."),
    ] = None,
    fixture_prompts: Annotated[bool, typer.Option("--fixture-prompts/--no-fixture-prompts")] = True,
    temperature: _Temperature = 0.8,
    top_k: _TopK = 40,
    max_new_tokens: _MaxNew = 48,
    seed: _Seed = 0,
) -> None:
    """Sample one output per prompt from a local EPOR checkpoint (exact tokens recorded)."""

    _cli().authorize_command("align teacher sample")
    from epor.alignment.fixtures import load_fixtures, prompt_lineage
    from epor.alignment.records import parent
    from epor.alignment.teachers import sample_teacher

    teacher = _model(checkpoint)
    store = _store()
    prompts: list[Any] = []
    try:
        if fixture_prompts:
            fixtures = load_fixtures()
            prompts += [
                (item.messages, (prompt_lineage(fixtures, item),)) for item in fixtures.prompts
            ]
        for identifier in prompt_record or []:
            prompts.append((store.get(identifier).messages, (parent(identifier),)))
    except Exception as exc:
        raise _failure(exc) from exc
    if not prompts:
        raise _refuse("no prompts: keep --fixture-prompts or name --prompt-record")
    sampling = _sampling(temperature, top_k, max_new_tokens, seed)
    written = _run(
        lambda cancel: sample_teacher(store, teacher, prompts, sampling, should_cancel=cancel)
    )
    _echo({"teacher": teacher.generator.model_dump(mode="json"), "examples": written})


@align_app.command("critique")
def align_critique(
    checkpoint: Annotated[Path, typer.Argument(help="Model run or checkpoint directory.")],
    example: Annotated[list[str], typer.Option("--example", help="Example record; repeatable.")],
    obligation: Annotated[
        list[str], typer.Option("--obligation", help="Covenant obligation key; repeatable.")
    ],
    max_new_tokens: _MaxNew = 48,
    seed: _Seed = 0,
) -> None:
    """Critique and revise examples against covenant obligations cited by key (candidates)."""

    _cli().authorize_command("align critique")
    from epor.alignment.critique import critique_and_revise

    model = _model(checkpoint)
    store = _store()
    sampling = _sampling(0.0, None, max_new_tokens, seed)
    written = _run(
        lambda cancel: critique_and_revise(
            store, model, example, obligation, sampling, should_cancel=cancel
        )
    )
    _echo({"candidates": written, "note": "critiques and revisions are candidates only"})


@align_app.command("eval-items")
def align_eval_items(
    checkpoint: Annotated[Path, typer.Argument(help="Model run or checkpoint directory.")],
    count: Annotated[int, typer.Option(min=1, max=600)] = 6,
    max_new_tokens: _MaxNew = 32,
    seed: _Seed = 0,
) -> None:
    """Write model-written G10 evaluation items as candidates for human review."""

    _cli().authorize_command("align eval-items")
    from epor.alignment.judge import generate_eval_items

    model = _model(checkpoint)
    store = _store()
    sampling = _sampling(0.0, None, max_new_tokens, seed)
    written = _run(
        lambda cancel: generate_eval_items(store, model, count, sampling, should_cancel=cancel)
    )
    _echo({"eval_items": written, "note": "candidates until a human accepts them"})


@align_app.command("judge")
def align_judge(
    checkpoint: Annotated[Path, typer.Argument(help="Judge run or checkpoint directory.")],
    subject: Annotated[Literal["preference", "revision", "eval-item"], typer.Option()],
    pair: Annotated[
        list[str] | None, typer.Option("--pair", help="FIRST:SECOND example records; repeatable.")
    ] = None,
    item: Annotated[list[str] | None, typer.Option("--item", help="Eval-item record.")] = None,
    all_unjudged: Annotated[
        bool,
        typer.Option("--all-unjudged", help="Judge every unjudged revision or eval item."),
    ] = False,
    obligation: Annotated[str, typer.Option(help="Covenant obligation key.")] = (
        "no-harmful-output"
    ),
    min_margin: Annotated[float, typer.Option(min=0.0, max=100.0)] = 0.05,
) -> None:
    """Record AI judge decisions; nothing they accept trains until humans audit the run."""

    _cli().authorize_command("align judge")
    from epor.alignment.judge import judge_eval_items, judge_pairs, unjudged_revisions

    judge = _model(checkpoint)
    store = _store()
    try:
        pairs = [tuple(entry.split(":", 1)) for entry in pair or []]
        if any(len(entry) != 2 for entry in pairs):
            raise ValueError("--pair expects FIRST:SECOND")
        items = list(item or [])
        if all_unjudged and subject == "revision":
            pairs += unjudged_revisions(store)
        elif all_unjudged and subject == "eval-item":
            judged = {
                record.inputs[0]
                for _, record in store.records("judgement")
                if record.subject == "eval-item"
            }
            items += [i for i in store.ids("eval-item") if i not in judged]
    except Exception as exc:
        raise _failure(exc) from exc

    def work(cancel: Any) -> Any:
        if subject == "eval-item":
            if not items:
                raise ValueError("no evaluation items to judge")
            return judge_eval_items(
                store, judge, items, min_margin=min_margin, should_cancel=cancel
            )
        if not pairs:
            raise ValueError("no pairs to judge")
        return judge_pairs(
            store,
            judge,
            pairs,  # type: ignore[arg-type]
            obligation=obligation,
            min_margin=min_margin,
            subject=subject,
            should_cancel=cancel,
        )

    result = _run(work)
    decisions = [store.get(item) for item in result["judgements"]]
    _echo(
        {
            **result,
            "accepted": sum(1 for decision in decisions if decision.accepted),
            "rejected": sum(1 for decision in decisions if not decision.accepted),
            "next": f"epor align audit sample {result['judging_run']}",
        }
    )


# --------------------------------------------------------------------- datasets


def _tokenizer(checkpoint: Path | None, tokenizer_id: str | None) -> Any:
    from epor.tokenizer import load_tokenizer

    if (checkpoint is None) == (tokenizer_id is None):
        raise _refuse("name exactly one of --checkpoint or --tokenizer-id")
    if checkpoint is not None:
        return _model(checkpoint).tokenizer
    try:
        return load_tokenizer(_cli()._data_root(), str(tokenizer_id))
    except Exception as exc:
        raise _failure(exc) from exc


def _build_dataset(
    kind: str,
    checkpoint: Path | None,
    tokenizer_id: str | None,
    include: list[str] | None,
    record: list[str] | None,
    validation_percent: int,
) -> None:
    from epor.alignment.audit import default_gate
    from epor.alignment.datasets import (
        build_preference_dataset,
        build_sft_dataset,
        describe_dataset,
        select_records,
    )

    tokenizer = _tokenizer(checkpoint, tokenizer_id)
    store = _store()
    try:
        ids = select_records(store, kind, include or [], record or [])
        if not ids:
            raise ValueError("no records selected: use --include and/or --record")
        build = build_sft_dataset if kind == "sft" else build_preference_dataset
        dataset_id, manifest = build(
            store,
            tokenizer,
            ids,
            validation_percent=validation_percent,
            gate=default_gate(_ledger(), store),
        )
    except Exception as exc:
        raise _failure(exc) from exc
    _echo(describe_dataset(dataset_id, manifest))


_Include = Annotated[
    list[str] | None, typer.Option("--include", help="Record group; repeatable (see help).")
]
_Record = Annotated[list[str] | None, typer.Option("--record", help="Record ID; repeatable.")]


@dataset_app.command("sft")
def dataset_sft(
    checkpoint: Annotated[Path | None, typer.Option(help="Take the tokenizer from here.")] = None,
    tokenizer_id: Annotated[str | None, typer.Option()] = None,
    include: _Include = None,
    record: _Record = None,
    validation_percent: Annotated[int, typer.Option(min=0, max=99)] = 10,
) -> None:
    """Build an SFT dataset (groups: fixtures, teacher-outputs, revisions, registered)."""

    _cli().authorize_command("align dataset")
    _build_dataset("sft", checkpoint, tokenizer_id, include, record, validation_percent)


@dataset_app.command("preference")
def dataset_preference(
    checkpoint: Annotated[Path | None, typer.Option(help="Take the tokenizer from here.")] = None,
    tokenizer_id: Annotated[str | None, typer.Option()] = None,
    include: _Include = None,
    record: _Record = None,
    validation_percent: Annotated[int, typer.Option(min=0, max=99)] = 10,
) -> None:
    """Build a preference dataset (groups: fixtures, ai-judge, registered, human)."""

    _cli().authorize_command("align dataset")
    _build_dataset("preference", checkpoint, tokenizer_id, include, record, validation_percent)


# --------------------------------------------------------------------- audits


@audit_app.command("sample")
def audit_sample(
    judging_run: Annotated[str, typer.Argument()],
    per_stratum: Annotated[int, typer.Option(min=1, max=500)] = 5,
    seed: _Seed = 0,
) -> None:
    """Queue a seeded sample of a run's accepted and rejected decisions for human audit."""

    cli = _cli()
    cli.authorize_command("align audit sample")
    from epor.alignment.audit import sample_run
    from epor.control.authority import LOCAL_CLI

    try:
        queued = sample_run(
            _ledger(),
            _store(),
            judging_run,
            per_stratum=per_stratum,
            seed=seed,
            actor_id=LOCAL_CLI.id,
        )
    except Exception as exc:
        raise _failure(exc) from exc
    _echo({"judging_run": judging_run, "queued": queued, "state": "pending human verdicts"})


@audit_app.command("list")
def audit_list(
    pending_only: Annotated[bool, typer.Option("--pending")] = False,
    judging_run: Annotated[str | None, typer.Option("--run")] = None,
) -> None:
    """List audit items with the material a reviewer needs."""

    _cli().authorize_command("align audit list")
    try:
        items = _ledger().items()
    except Exception as exc:
        raise _failure(exc) from exc
    _echo(
        [
            item.view()
            for item in items
            if (not pending_only or item.verdict is None)
            and (judging_run is None or item.judging_run == judging_run)
        ]
    )


@audit_app.command("verdict")
def audit_verdict(
    item_id: Annotated[str, typer.Argument()],
    verdict: Annotated[Literal["accept", "reject"], typer.Argument()],
    note: Annotated[str, typer.Option(help="Optional reviewer note (at most 2,000 chars).")] = "",
) -> None:
    """Record the owner's verdict on one audit item (the only way an item is audited)."""

    _cli().authorize_command("align audit verdict")
    from epor.control.authority import LOCAL_CLI

    try:
        record = _ledger().record_verdict(item_id, verdict, actor=LOCAL_CLI, note=note)
    except Exception as exc:
        raise _failure(exc) from exc
    _echo({"item": item_id, "verdict": verdict, "record_sha256": record["record_sha256"]})


@audit_app.command("calibration")
def audit_calibration(judging_run: Annotated[str, typer.Argument()]) -> None:
    """Store and print a run's judge calibration and judge-human disagreement rate."""

    _cli().authorize_command("align audit calibration")
    from epor.alignment.audit import calibration
    from epor.alignment.monitors import store_report

    store = _store()
    try:
        report = {
            "schema_version": 1,
            "kind": "calibration",
            **calibration(_ledger(), judging_run, store=store),
        }
        report_id = store_report(store, "calibration", report)
    except Exception as exc:
        raise _failure(exc) from exc
    _echo({"report_id": report_id, **report})


# --------------------------------------------------------------------- evaluation


@align_app.command("monitor")
def align_monitor(
    policy: Annotated[Path, typer.Argument(help="Preference-trained run or checkpoint.")],
    reference: Annotated[Path, typer.Option(help="Frozen reference run or checkpoint.")],
    dataset: Annotated[str, typer.Option(help="Preference dataset (validation split).")],
    beta: Annotated[float, typer.Option(min=1e-6, max=10.0)] = 0.1,
    max_new_tokens: _MaxNew = 32,
) -> None:
    """Measure KL, proxy-vs-audited agreement, and length drift (exit 1 at abort)."""

    _cli().authorize_command("align monitor")
    from epor.alignment.fixtures import load_fixtures
    from epor.alignment.monitors import audited_pairs, held_out_pairs, overoptimization_report

    policy_model, reference_model = _model(policy), _model(reference)
    store = _store()
    try:
        held_out = held_out_pairs(store, dataset)
        audited = audited_pairs(store, _ledger())
        prompts = [item.messages for item in load_fixtures().prompts]
        prompts += [messages for messages, _, _ in held_out]
    except Exception as exc:
        raise _failure(exc) from exc
    report_id, report = _run(
        lambda cancel: overoptimization_report(
            store,
            policy_model,
            reference_model,
            prompts=prompts,
            held_out_pairs=held_out,
            audited_pairs=audited,
            beta=beta,
            max_new_tokens=max_new_tokens,
            should_cancel=cancel,
        )
    )
    _echo({"report_id": report_id, "status": report["status"], "monitors": report["monitors"]})
    if report["status"] == "abort":
        typer.echo("overoptimization monitor at abort threshold (hypothesis)", err=True)
        raise typer.Exit(code=1)


@align_app.command("g10")
def align_g10(
    before: Annotated[Path, typer.Option(help="Checkpoint before the alignment step.")],
    after: Annotated[Path, typer.Option(help="Checkpoint after the alignment step.")],
    tolerance: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.0,
    max_new_tokens: _MaxNew = 32,
) -> None:
    """Compare G10 behaviour before and after alignment (exit 1 blocked, 4 escalated)."""

    _cli().authorize_command("align g10")
    from epor.alignment.g10 import run_g10
    from epor.control.authority import LOCAL_CLI

    before_model, after_model = _model(before), _model(after)
    store, ledger = _store(), _ledger()
    report_id, report = _run(
        lambda cancel: run_g10(
            store,
            before_model,
            after_model,
            ledger=ledger,
            actor_id=LOCAL_CLI.id,
            tolerance=tolerance,
            max_new_tokens=max_new_tokens,
            should_cancel=cancel,
        )
    )
    _echo(
        {
            "report_id": report_id,
            "gate": report["gate"],
            "categories": report["categories"],
            "regressed_categories": report["regressed_categories"],
            "failing_items": report["failing_items"],
            "claim": report["claim"],
        }
    )
    if report["gate"] == "blocked":
        typer.echo("G10 regression: the aligned checkpoint fails closed", err=True)
        raise typer.Exit(code=1)
    if report["gate"] == "escalated":
        typer.echo("G10 failures queued for human review (epor align audit list)", err=True)
        raise typer.Exit(code=_ESCALATED_EXIT)


@align_app.command("reports")
def align_reports() -> None:
    """List alignment records, datasets, and stored reports by ID."""

    _cli().authorize_command("align reports")
    from epor.alignment.summary import alignment_summary

    try:
        summary = alignment_summary(_cli()._data_root(), _ledger())
    except Exception as exc:
        raise _failure(exc) from exc
    _echo(summary)


# --------------------------------------------------------------------- training


def train_sft(
    init: Annotated[
        Path | None,
        typer.Argument(help="Parent run or checkpoint (omit with --recipe for a fresh student)."),
    ] = None,
    dataset: Annotated[str, typer.Option(help="Verified SFT chat dataset ID.")] = "",
    output: Annotated[Path, typer.Option("--output", help="New run directory.")] = Path("runs/sft"),
    max_steps: Annotated[int, typer.Option(min=1, max=100_000)] = 50,
    recipe: Annotated[
        Path | None, typer.Option(help="Fresh student recipe (response/logit distillation).")
    ] = None,
    tokenized_id: Annotated[
        str | None, typer.Option(help="Tokenized build fixing the tokenizer (with --recipe).")
    ] = None,
    logit_teacher: Annotated[
        list[Path] | None,
        typer.Option("--logit-teacher", help="EPOR teacher checkpoint for logit distillation."),
    ] = None,
    distill_weight: Annotated[float, typer.Option(min=1e-6, max=10.0)] = 1.0,
    distill_temperature: Annotated[float, typer.Option(min=0.25, max=10.0)] = 1.0,
    seq_len: Annotated[int, typer.Option(min=1, max=8_192)] = 128,
    tokens_per_batch: Annotated[int, typer.Option(min=1, max=1_048_576)] = 1_024,
    micro_batch_size: Annotated[int | None, typer.Option(min=1)] = None,
    precision: Annotated[Literal["bf16", "fp32"], typer.Option()] = "bf16",
    strategy: Annotated[Literal["none", "ddp", "fsdp2"], typer.Option()] = "none",
    world_size: Annotated[int, typer.Option(min=1, max=8)] = 1,
    learning_rate: Annotated[float, typer.Option(min=1e-8, max=1.0)] = 1e-4,
    warmup_steps: Annotated[int, typer.Option(min=0, max=100_000)] = 0,
    weight_decay: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.0,
    data_seed: _Seed = 23,
    model_seed: _Seed = 17,
    checkpoint_every: Annotated[int, typer.Option(min=1, max=100_000)] = 50,
    keep_checkpoints: Annotated[int, typer.Option(min=1, max=100)] = 2,
    validation: Annotated[bool, typer.Option("--validation/--no-validation")] = True,
) -> None:
    """Supervised fine-tuning and distillation on the training kernel (assistant-only loss).

    Resume an interrupted run with ``epor train kernel-resume``.
    """

    cli = _cli()
    cli.authorize_command("train sft")
    from epor.models.config import load_model_config, model_config_from_mapping
    from epor.training import OptimizerSettings, PackedTrainingSettings
    from epor.training.kernel import LogitDistillation, resolve_model_config
    from epor.training.run_checkpoint import read_kernel_checkpoint

    if not dataset:
        raise _refuse("--dataset is required")
    if (init is None) == (recipe is None):
        raise _refuse("name a parent checkpoint or a --recipe for a fresh student, not both")
    if (strategy == "none") != (world_size == 1):
        raise _refuse("strategy 'none' runs one process and ddp/fsdp2 need --world-size >= 2")
    project_root, destination = cli._reference_path(output, label="output path", must_exist=False)
    try:
        init_ref = None
        if init is not None:
            init_ref = _ref(init)
            trainer = read_kernel_checkpoint(init_ref.path)
            model = resolve_model_config(model_config_from_mapping(trainer["model_config"]))
            build = PackedTrainingSettings.model_validate(trainer["settings"]).tokenized_id
        else:
            assert recipe is not None
            if tokenized_id is None:
                raise ValueError("--tokenized-id is required with --recipe")
            recipe_path = cli._reference_path(recipe, label="recipe path", must_exist=True)[1]
            model = resolve_model_config(load_model_config(recipe_path))
            build = tokenized_id
        distillation = None
        if logit_teacher:
            distillation = LogitDistillation(
                teachers=tuple(_ref(path) for path in logit_teacher),
                weight=distill_weight,
                temperature=distill_temperature,
            )
        settings = PackedTrainingSettings(
            tokenized_id=build,
            chat_dataset_id=dataset,
            init_checkpoint=init_ref,
            distillation=distillation,
            seq_len=seq_len,
            tokens_per_batch=tokens_per_batch,
            micro_batch_size=micro_batch_size,
            data_seed=data_seed,
            model_seed=model_seed,
            precision=precision,
            checkpoint_every=checkpoint_every,
            keep_checkpoints=keep_checkpoints,
            validation_split="validation" if validation else None,
            optimizer=OptimizerSettings(
                learning_rate=learning_rate,
                warmup_steps=warmup_steps,
                total_steps=max_steps,
                weight_decay=weight_decay,
            ),
        )
    except typer.Exit:
        raise
    except Exception as exc:
        raise _failure(exc) from exc
    summary = cli._run_kernel_command(
        model,
        destination,
        settings,
        max_steps=max_steps,
        resume_from=None,
        strategy=strategy,
        world_size=world_size,
        launch_timeout=6 * 3600.0,
        project_root=project_root,
    )
    cli._echo_json(
        {
            **summary,
            "objective": "logit-distillation" if distillation else "sft",
            "chat_dataset_id": dataset,
            "scope": "proxy-scale local fine-tune; not evidence of alignment or safety",
        }
    )


def train_preference(
    init: Annotated[Path, typer.Argument(help="SFT run or checkpoint the policy starts from.")],
    dataset: Annotated[str, typer.Option(help="Verified preference dataset ID.")],
    output: Annotated[Path, typer.Option("--output", help="Run directory.")],
    max_steps: Annotated[int, typer.Option(min=1, max=100_000)] = 50,
    reference: Annotated[
        Path | None, typer.Option(help="Frozen reference (default: the starting checkpoint).")
    ] = None,
    resume: Annotated[
        Path | None, typer.Option(help="Continue a DPO run from this checkpoint.")
    ] = None,
    beta: Annotated[float, typer.Option(min=1e-3, max=10.0)] = 0.1,
    pairs_per_step: Annotated[int, typer.Option(min=1, max=64)] = 4,
    learning_rate: Annotated[float, typer.Option(min=1e-8, max=1.0)] = 1e-4,
    precision: Annotated[Literal["bf16", "fp32"], typer.Option()] = "fp32",
    seed: _Seed = 29,
    checkpoint_every: Annotated[int, typer.Option(min=1, max=100_000)] = 25,
) -> None:
    """DPO baseline against a frozen reference (exit 1 at the monitor's abort threshold)."""

    cli = _cli()
    cli.authorize_command("train preference")
    from epor.alignment.monitors import load_monitor_config
    from epor.alignment.preference import DpoMonitor, DpoSettings
    from epor.alignment.preference import train_preference as run_dpo
    from epor.training import OptimizerSettings

    project_root, destination = cli._reference_path(output, label="output path", must_exist=False)
    resume_dir = _checkpoint(resume) if resume is not None else None
    try:
        policy_ref = _ref(init)
        limits = load_monitor_config().dpo_step_logratio
        settings = DpoSettings(
            dataset_id=dataset,
            policy_init=policy_ref,
            reference=_ref(reference) if reference is not None else policy_ref,
            beta=beta,
            pairs_per_step=pairs_per_step,
            seed=seed,
            precision=precision,
            optimizer=OptimizerSettings(
                learning_rate=learning_rate, weight_decay=0.0, total_steps=max_steps
            ),
            monitor=DpoMonitor(logratio_alert=limits.alert, logratio_abort=limits.abort),
            checkpoint_every=checkpoint_every,
        )
    except Exception as exc:
        raise _failure(exc) from exc

    def on_step(step: int, total: int, loss: float) -> None:
        typer.echo(f"step {step}/{total} dpo_loss {loss:.6f}", err=True)

    def work(cancel: Any) -> Any:
        return run_dpo(
            cli._data_root(),
            destination,
            settings,
            max_steps=max_steps,
            resume_from=resume_dir,
            project_root=project_root,
            should_cancel=cancel,
            on_step=on_step,
        )

    from epor.alignment.preference import MonitorAbortError

    try:
        stop = threading.Event()
        with cli._StopRequests(stop.set):
            result = work(stop.is_set)
    except MonitorAbortError as exc:
        cli.console.print(f"[red]stopped by monitor[/red]: {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        raise _failure(exc) from exc
    cli._echo_json(
        {
            **result.model_dump(mode="json"),
            "scope": "proxy-scale DPO baseline; not evidence of alignment or safety",
        }
    )


def register_alignment_commands(app: typer.Typer, train_app: typer.Typer) -> None:
    app.add_typer(align_app, name="align")
    train_app.command("sft")(train_sft)
    train_app.command("preference")(train_preference)


__all__ = ["align_app", "register_alignment_commands"]
