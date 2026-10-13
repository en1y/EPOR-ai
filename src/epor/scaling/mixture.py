"""Data-mixture ablations: one small recipe, several domain mixtures, per-domain validation.

A mixture is a set of target token shares over the data-engine domain labels
of one tokenized build (``DatasetDocument.domain``).  The smallest correct
materialization that fits the existing pipeline is a *training subset*: the
kernel's ``train_mixture`` setting selects a deterministic, non-repeating
subset of the build's training documents whose token shares match the weights
(:mod:`epor.training.mixture` documents the rule).  No data-engine build,
tokenizer, or token shard is rewritten, so every mixture shares one tokenized
build, one tokenizer, and one validation split, and its lineage is the build
plus the mixture definition and the selection hash.

:func:`run_mixture_ablation` trains the same model for the same number of
steps on each mixture (matched training tokens; the subsets differ in unique
tokens, which the report records as epochs over the subset), then measures
validation main loss on each domain's full validation documents and on the
whole validation split.  Points are resumable and cancellable exactly like
sweep points (:mod:`epor.scaling.sweep`).

Whether one mixture beats another is decided only by these measured losses;
the roadmap's initial proxy mixture (55/25/15/5) is a starting point, not a
result.  Losses are proxy-scale measurements.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from epor.models.config import load_model_config

from .ladder import PROJECT_ROOT
from .report import ScalingReport, ScalingScope, seal_scaling_report
from .sweep import point_id, read_json, train_point, write_point


class NamedMixture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mixture_id: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")]
    weights: dict[str, float]
    seed: int = Field(default=0, ge=0)

    def domain_mixture(self) -> Any:
        from epor.training.mixture import DomainMixture

        return DomainMixture(weights=self.weights, seed=self.seed)


class MixturePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1)
    mixtures: tuple[NamedMixture, ...] = Field(min_length=1)
    max_steps: int = Field(ge=1)
    evaluation_split: str = "validation"
    evaluation_max_steps: int | None = Field(default=None, ge=1)
    training: Any

    @field_validator("mixtures")
    @classmethod
    def unique_mixtures(cls, value: tuple[NamedMixture, ...]) -> tuple[NamedMixture, ...]:
        if len({item.mixture_id for item in value}) != len(value):
            raise ValueError("mixture IDs must be unique")
        return value

    @field_validator("evaluation_split")
    @classmethod
    def known_split(cls, value: str) -> str:
        if value not in {"train", "validation", "test"}:
            raise ValueError("evaluation_split must be train, validation, or test")
        return value

    @field_validator("training")
    @classmethod
    def training_settings(cls, value: Any) -> Any:
        from epor.training.kernel import PackedTrainingSettings

        settings = (
            value
            if isinstance(value, PackedTrainingSettings)
            else PackedTrainingSettings.model_validate(value)
        )
        if settings.train_mixture is not None:
            raise ValueError("the training template must not set train_mixture")
        return settings

    def payload(self) -> dict[str, Any]:
        return {
            **self.model_dump(mode="json", exclude={"training"}),
            "training": self.training.model_dump(mode="json"),
        }


def _domain_documents(data_root: Path, tokenized_id: str, split: Any) -> dict[str, list[str]]:
    from epor.training.mixture import document_domains
    from epor.training.shard_data import ShardCorpus

    corpus = ShardCorpus.open(data_root, tokenized_id, split, verify=False)
    domains = document_domains(data_root, corpus.manifest.dataset_id, corpus.manifest.build_id)
    grouped: dict[str, list[str]] = {}
    for document_id in corpus.document_ids:
        grouped.setdefault(domains[document_id], []).append(document_id)
    return dict(sorted(grouped.items()))


def _evaluate(
    run_dir: Path,
    data_root: Path,
    split: Any,
    documents: dict[str, list[str]],
    max_steps: int | None,
    should_cancel: Callable[[], bool] | None,
) -> dict[str, Any]:
    from epor.training.kernel import evaluate_packed_checkpoint
    from epor.training.run_checkpoint import latest_checkpoint

    checkpoint = latest_checkpoint(run_dir)
    if checkpoint is None:
        raise RuntimeError(f"mixture run has no checkpoint: {run_dir}")

    def one(ids: list[str] | None) -> dict[str, Any]:
        result = evaluate_packed_checkpoint(
            checkpoint,
            data_root,
            split=split,
            max_steps=max_steps,
            should_cancel=should_cancel,
            document_ids=ids,
        )
        evaluation = result.evaluation
        return {
            "main_loss": evaluation.main_loss,
            "perplexity": evaluation.perplexity,
            "target_tokens": evaluation.target_tokens,
            "documents": len(ids) if ids is not None else None,
        }

    by_domain = {domain: one(ids) for domain, ids in documents.items()}
    # A capped evaluation pass walks documents in a fixed order, so evaluating
    # the whole split under a cap would score only the first domain(s). The
    # overall figure is therefore the target-token-weighted mean of the
    # per-domain losses, which equals the global mean exactly when uncapped.
    target_tokens = sum(item["target_tokens"] for item in by_domain.values())
    main_loss = (
        sum(item["main_loss"] * item["target_tokens"] for item in by_domain.values())
        / target_tokens
        if target_tokens
        else float("nan")
    )
    overall = {
        "main_loss": main_loss,
        "perplexity": math.exp(main_loss) if target_tokens else float("nan"),
        "target_tokens": target_tokens,
        "documents": sum(item["documents"] or 0 for item in by_domain.values()),
    }
    return {"overall": overall, "by_domain": by_domain}


def run_mixture_ablation(
    plan: MixturePlan,
    *,
    data_root: str | Path,
    output_dir: str | Path,
    project_root: str | Path = PROJECT_ROOT,
    should_cancel: Callable[[], bool] | None = None,
    on_mixture: Callable[[int, int, dict[str, Any]], None] | None = None,
    scope: ScalingScope = "proxy-scale",
) -> ScalingReport:
    """Train and evaluate one model per mixture; return the sealed ablation report."""

    from epor.training.loop import TrainingCancelled
    from epor.training.manifest import hardware_snapshot, utc_now
    from epor.training.mixture import select_mixture_documents
    from epor.training.runner import build_vocab_size
    from epor.training.shard_data import ShardCorpus

    from .ladder import derive_rung

    root = Path(project_root).resolve()
    data = Path(data_root).resolve()
    output = Path(output_dir).resolve()
    tokenized_id = plan.training.tokenized_id
    model_path = Path(plan.model)
    recipe = load_model_config(model_path if model_path.is_absolute() else root / model_path)
    config = derive_rung(recipe, build_vocab_size(data, tokenized_id))
    split = plan.evaluation_split
    documents = _domain_documents(data, tokenized_id, split)
    if not documents:
        raise ValueError(f"the {split} split has no documents to evaluate")
    template = plan.training.model_copy(
        update={
            "keep_checkpoints": 1,
            "checkpoint_every": min(plan.training.checkpoint_every, plan.max_steps),
        }
    )
    train_corpus = ShardCorpus.open(data, tokenized_id, template.train_split, verify=False)
    manifest = train_corpus.manifest

    results: list[dict[str, Any]] = []
    measured_runs: list[dict[str, Any]] = []
    for index, named in enumerate(plan.mixtures):
        if should_cancel is not None and should_cancel():
            raise TrainingCancelled("mixture ablation cancellation requested")
        mixture = named.domain_mixture()
        settings = type(template).model_validate(
            {**template.model_dump(mode="json"), "train_mixture": mixture.model_dump(mode="json")}
        )
        selection = select_mixture_documents(
            data, tokenized_id, mixture, split=settings.train_split
        )
        identifier = point_id(config, settings, plan.max_steps)
        point_file = output / "points" / f"{identifier}.json"
        if point_file.is_file():
            stored = read_json(point_file)
            result, measured = stored["result"], stored["measured"]
        else:
            run_dir_root = output / "runs" / identifier
            outcome, run_measured = train_point(
                config,
                settings,
                plan.max_steps,
                run_dir_root,
                data_root=data,
                project_root=root,
                should_cancel=should_cancel,
            )
            evaluation = _evaluate(
                Path(run_measured["run_dir"]),
                data,
                split,
                documents,
                plan.evaluation_max_steps,
                should_cancel,
            )
            trained_pairs = int(outcome.get("target_tokens_seen") or 0)
            result = {
                "mixture_id": named.mixture_id,
                "point_id": identifier,
                "mixture": mixture.model_dump(mode="json"),
                "mixture_sha256": mixture.sha256,
                "selection": selection.summary(),
                "epochs_over_subset": (
                    trained_pairs / selection.selected_pairs if selection.selected_pairs else None
                ),
                "final_main_loss": outcome["final_main_loss"],
                "tokens_seen": outcome["tokens_seen"],
                "validation": evaluation,
                "loss_curve": outcome["loss_curve"],
            }
            measured = {"point_id": identifier, **run_measured}
            write_point(point_file, {"result": result, "measured": measured})
        results.append(result)
        measured_runs.append(measured)
        if on_mixture is not None:
            on_mixture(index + 1, len(plan.mixtures), result)

    return seal_scaling_report(
        kind="mixture-ablation",
        scope=scope,
        inputs={
            "plan": plan.payload(),
            "model_config_sha256": config.sha256,
            "recipe_config_sha256": recipe.sha256,
            "lineage": {
                "tokenized_id": tokenized_id,
                "tokenizer_id": manifest.tokenizer_id,
                "dataset_id": manifest.dataset_id,
                "build_id": manifest.build_id,
            },
            "evaluation_documents": {domain: len(ids) for domain, ids in documents.items()},
            "materialization": (
                "kernel train_mixture: deterministic non-repeating domain-weighted subset of "
                "the training split; validation is never re-weighted"
            ),
        },
        results={"mixtures": results},
        warnings=[
            "Proxy-scale ablation: per-domain losses compare mixtures for this small model "
            "and build only.",
            "Mixtures are matched on training tokens, not on unique tokens; see "
            "epochs_over_subset.",
        ],
        measured={
            "recorded_at": utc_now(),
            "host": hardware_snapshot(),
            "output_dir": str(output),
            "runs": measured_runs,
        },
    )


def mixture_summary(report: ScalingReport) -> list[dict[str, Any]]:
    """``[{mixture_id, by_domain: {domain: loss}, overall}]`` for display."""

    if report.kind != "mixture-ablation":
        raise ValueError("not a mixture-ablation report")
    return [
        {
            "mixture_id": item["mixture_id"],
            "overall": item["validation"]["overall"]["main_loss"],
            "by_domain": {
                domain: values["main_loss"]
                for domain, values in item["validation"]["by_domain"].items()
            },
        }
        for item in report.results["mixtures"]
    ]


__all__ = ["MixturePlan", "NamedMixture", "mixture_summary", "run_mixture_ablation"]
