"""Every action EPOR can execute, and its reviewed covenant standing.

This is the single registry behind both entry points: the control plane's job
admission and the CLI.  Neither has its own table, so an action cannot be
governed on one path and ungoverned on the other.

Assessments are declared and reviewed here rather than inferred, which is only
defensible because the v0.0.2 action space is a closed list of local,
operator-directed work.  Its network listeners are fixed to numeric loopback
addresses, and its job work remains offline and resource-bounded.  An action
with no declaration resolves to escalation: adding executable work to EPOR
requires stating its covenant standing first, and forgetting to fails closed
rather than open.

The declarations are reviewed judgements, not measurements.  None of them make
the executed work safe on their own.
"""

from __future__ import annotations

from typing import Protocol

from .safety import (
    PRIORITIES,
    Covenant,
    PrincipleAssessment,
    Priority,
    ProposedAction,
    Resolution,
    resolve,
)


def _declared(
    summary: str,
    *,
    people: str,
    direction: str,
    system: str,
    protects: tuple[Priority, ...] = (),
) -> ProposedAction:
    """Declare an action whose every principle is assessed as satisfied."""

    rationales = (people, direction, system)
    return ProposedAction(
        summary=summary,
        protects=protects,
        assessments=tuple(
            PrincipleAssessment(
                priority=priority,
                status="satisfied",
                confidence=0.99,
                rationale=rationale,
            )
            for priority, rationale in zip(PRIORITIES, rationales, strict=True)
        ),
    )


_LOCAL_AND_OFFLINE = (
    "runs locally under the operator's own account, reaches no person other than "
    "that operator, and emits no personal data"
)
_OPERATOR_REQUEST = (
    "invoked directly by the operator who owns the machine, within the reviewed action registry"
)

DECLARED_ACTIONS: dict[str, ProposedAction] = {
    "system_probe": _declared(
        "read local capability information",
        people=f"read-only inspection of this machine; {_LOCAL_AND_OFFLINE}",
        direction=_OPERATOR_REQUEST,
        system="read-only and side-effect free",
    ),
    "research_sync": _declared(
        "download allowlisted research sources into ignored local paths",
        people=(
            "fetches published research from exact allowlisted hosts into paths "
            "excluded from Git; extraction is inert and stores no personal data"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; the source list is reviewed configuration and no "
            "caller may supply a URL, so text inside a fetched document can never "
            "become an instruction"
        ),
        system="bounded sizes, atomic replacement, and checksum verification",
    ),
    "tiny_train": _declared(
        "train the tiny reference model on a bounded local corpus",
        people=(
            f"a 10-50M parameter fixture model trained offline; {_LOCAL_AND_OFFLINE}, "
            "and the result is neither released nor served"
        ),
        direction=f"{_OPERATOR_REQUEST}; the corpus is capped and contained below the roots",
        system=(
            "fixed model, corpus, checkpoint, batch-token, and step limits keep the reference "
            "path CPU-safe; control jobs add checkpoint-boundary cooperative cancellation"
        ),
    ),
    "tiny_eval": _declared(
        "evaluate a tiny reference checkpoint on a bounded local corpus",
        people=f"offline measurement of a fixture checkpoint; {_LOCAL_AND_OFFLINE}",
        direction=f"{_OPERATOR_REQUEST}; the checkpoint is contained below the configured root",
        system=(
            "bounded batches and inputs; read-only over the checkpoint, with cooperative "
            "cancellation in control jobs"
        ),
    ),
    "generate": _declared(
        "greedily generate text from a tiny reference checkpoint",
        people=(
            "the only declared action whose output a person reads. A 10-50M fixture "
            "model over a debug tokenizer produces text with no informational "
            "content, printed locally to the operator who asked for it and never "
            "presented as fact. This rationale does not survive a trained model: a "
            "released checkpoint requires output classification under "
            "no-harmful-output before this declaration can be reused"
        ),
        direction=f"{_OPERATOR_REQUEST}; the prompt comes from the operator's own command line",
        system="bounded checkpoint and token count, and read-only over the checkpoint",
    ),
    "control_api_start": _declared(
        "start the typed control API on numeric loopback",
        people=(
            "binds only to numeric loopback on the documented trusted single-user machine and "
            "admits only separately declared job types; it accepts neither shell commands nor "
            "arbitrary URLs"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; every submitted job is independently authorized before its "
            "specification is parsed"
        ),
        system=(
            "fixed loopback binding, exact local UI origin, durable SQLite state, and unrestricted "
            "shutdown"
        ),
    ),
    "job_worker_start": _declared(
        "start the local allowlisted job worker",
        people=(
            "dispatches only typed jobs whose own declarations are re-resolved immediately before "
            "execution and exposes no network listener"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; queued work carries its admission audit and is independently "
            "authorized again at dispatch"
        ),
        system=(
            "closed handler map, bounded polling, cooperative cancellation, and unrestricted "
            "shutdown"
        ),
    ),
    "local_ui_start": _declared(
        "start the browser console on its fixed loopback origin",
        people=(
            "uses a fixed loopback origin on the trusted single-user machine and can submit typed "
            "control jobs but no shell command or arbitrary URL"
        ),
        direction=f"{_OPERATOR_REQUEST}; browser actions remain subject to API job admission",
        system="fixed loopback origin, pinned UI dependencies, and unrestricted shutdown",
    ),
    "config_validate": _declared(
        "strictly validate a tracked model recipe without allocating weights",
        people=f"reads tracked configuration and allocates nothing; {_LOCAL_AND_OFFLINE}",
        direction=f"{_OPERATOR_REQUEST}; the recipe path resolves beneath the project root",
        system="read-only, offline, and side-effect free apart from printed output",
    ),
    "research_verify": _declared(
        "verify local research files against their tracked checksums",
        people=(
            "recomputes digests over already-downloaded files; it fetches nothing "
            f"and {_LOCAL_AND_OFFLINE}"
        ),
        direction=f"{_OPERATOR_REQUEST}; the source list is reviewed configuration",
        system="read-only over ignored local paths and safe to repeat",
    ),
    "research_index": _declared(
        "extract text from verified local research files into ignored paths",
        people=(
            "extraction is inert: no script, macro, external reference, or active "
            "content is evaluated, and text inside an extracted document stays "
            "data rather than instruction"
        ),
        direction=f"{_OPERATOR_REQUEST}; only catalog-listed local files are read",
        system="bounded byte limits, atomic replacement, and Git-excluded output",
    ),
    "data_ingest": _declared(
        "register and ingest reviewed local data into immutable private layers",
        people=(
            "requires a rights and removal registration before reading local bytes; PII and "
            "credential values are removed, unsafe content is quarantined, and ordinary logs "
            "retain only finding kinds and counts"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; every input is a contained local file and text inside it is "
            "always data, never an instruction"
        ),
        system=(
            "streaming size limits, private content-addressed layers, immutable provenance, "
            "cooperative cancellation, and no network acquisition"
        ),
    ),
    "data_build": _declared(
        "build deterministic filtered dataset splits and a dataset card",
        people=(
            "uses only training-eligible reviewed registrations after sensitive-value removal, "
            "quarantine, tombstones, and duplicate controls; it publishes nothing"
        ),
        direction=f"{_OPERATOR_REQUEST}; build policy is typed and resource-bounded",
        system=(
            "immutable parent hashes, global deduplication before stable split assignment, "
            "idempotent outputs, and cooperative cancellation"
        ),
    ),
    "data_remove": _declared(
        "record an immutable removal tombstone for all future dataset builds",
        people=(
            "honors a reviewed removal direction without exposing the removed content and "
            "prevents the target entering future builds"
        ),
        direction=f"{_OPERATOR_REQUEST}; the target and rationale are explicit and auditable",
        system="append-only tombstones preserve lineage without mutating historical evidence",
        protects=(1, 2),
    ),
    "data_audit": _declared(
        "verify local data-layer hashes and summarize provenance state",
        people=f"reads only local metadata and content hashes; {_LOCAL_AND_OFFLINE}",
        direction=f"{_OPERATOR_REQUEST}; the audit changes no dataset record",
        system="read-only integrity verification over the private data root",
    ),
    "tokenizer_train": _declared(
        "train an original byte-level BPE tokenizer from one dataset build's train split",
        people=(
            "reads only the train split of a local build, which already holds only "
            "training-eligible reviewed registrations after sensitive-value removal, quarantine, "
            "and duplicate controls, and refuses a build with documents removed since it was "
            "made; it publishes nothing and the artifact stays in the private data root"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; vocabulary and frequency settings are typed and bounded, and "
            "corpus text is only ever counted, never interpreted as an instruction"
        ),
        system=(
            "single-threaded deterministic training, a content-addressed immutable artifact "
            "pinned to the frozen reserved-ID spec, and cooperative cancellation"
        ),
    ),
    "tokenizer_shard": _declared(
        "encode one dataset build into content-addressed token shards",
        people=(
            "re-applies every current removal tombstone so removed documents never enter shards, "
            "writes only to the private data root, and publishes nothing"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; the build and tokenizer are hash-verified and text cannot "
            "forge reserved control tokens, because reserved IDs appear only as document "
            "boundaries"
        ),
        system=(
            "staged atomic publication, immutable content-addressed shards and indexes, "
            "idempotent rebuilds, bounded shard sizes, and cooperative cancellation"
        ),
    ),
    "tokenizer_evaluate": _declared(
        "measure tokenizer fertility and byte round-trip on one dataset split",
        people=(
            "reads local split text and records only aggregate counts and ratios, never document "
            f"text; tombstoned documents are excluded; {_LOCAL_AND_OFFLINE}"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; the report names its split, flags in-sample measurement, and "
            "states that it is not a target-scale result"
        ),
        system=(
            "read-only over verified artifacts, an immutable content-addressed report, and "
            "cooperative cancellation"
        ),
    ),
    "kernel_train": _declared(
        "pretrain a bounded dense model on one verified tokenized build",
        people=(
            "reads only token shards of a tokenized build that verifies against its content "
            "hashes and was cut from reviewed, training-eligible data after removal and "
            "tombstone controls; the model is a local proxy-scale research artifact of at most "
            "50M parameters that is neither released, served, nor used to generate text here, "
            f"and nothing is published; {_LOCAL_AND_OFFLINE}"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; every setting is typed and bounded by the kernel policy "
            "(dense only, at most 50M parameters, 100k steps, and 1,048,576 tokens per step), "
            "token IDs are only ever counted and embedded, never interpreted as instructions, "
            "and no surface accepts a fault-injection plan"
        ),
        system=(
            "local CPU only with no paid compute; distributed ranks are spawned on this machine "
            "and listen on loopback alone; checkpoints are hashed and promoted atomically, "
            "resume refuses any change to the trajectory, and cooperative cancellation "
            "checkpoints the last completed step without ever being gated"
        ),
    ),
    "kernel_evaluate": _declared(
        "measure a kernel checkpoint's loss on one tokenized split",
        people=(
            "reads a hash-verified local checkpoint and one split of a verified tokenized build "
            "and records only aggregate losses, perplexity, and token counts, never text; "
            f"{_LOCAL_AND_OFFLINE}"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; the report names its checkpoint and split and is labelled a "
            "proxy-scale measurement rather than a validated result"
        ),
        system=(
            "read-only over the checkpoint and shards, single process, bounded batches, an "
            "atomically written report, and cooperative cancellation"
        ),
    ),
    "eval_suite": _declared(
        "score a kernel checkpoint on a versioned evaluation suite",
        people=(
            "scores a hash-verified local proxy-scale checkpoint on project-authored items or on "
            "local benchmark files whose registration allows evaluation and that no tombstone "
            "removes; generated text is only compared against references and stored in a local "
            "report, never shown as advice or acted on, and nothing is published or downloaded; "
            f"{_LOCAL_AND_OFFLINE}"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; item and benchmark text is always data and never an "
            "instruction, code tasks are skipped unless code execution is separately "
            "authorized, and every metric carries its harness-fixture or proxy-scale scope "
            "and the report's claim boundary"
        ),
        system=(
            "read-only over the checkpoint, builds, and benchmark files; single process on the "
            "CPU, a content-addressed report written once, safe answer parsing without eval, and "
            "cooperative cancellation between and inside tasks"
        ),
    ),
    "eval_code_execution": _declared(
        "execute model-generated code candidates in the evaluation sandbox",
        people=(
            "runs untrusted model-generated Python only for code tasks, each candidate in a "
            "fresh python -I -S -B process with a private 0700 working directory deleted "
            "afterwards, a stripped environment that inherits no secret, /dev/null stdin, "
            "rlimits on CPU, address space, file size, open files, and processes, no core "
            "dumps, and a wall-clock kill of its process group; where the kernel permits it "
            "the candidate also gets an unprivileged user and network namespace with no "
            "network, and Landlock confining writes to its working directory. These layers "
            "are defense in depth against accidents, not a security boundary: reads are not "
            "confined and there is no seccomp filter, cgroup, PID, or mount namespace. When "
            "the network namespace is unavailable the run is refused unless the operator "
            "separately opts in to unisolated network with its own flag, and the layers "
            "actually achieved are recorded in the report"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; off by default everywhere, it is a separate declared action "
            "so a delegation scoped to evaluation never implies it, and it is available only "
            "from the owner's own shell: the control-plane job API cannot request it"
        ),
        system=(
            "candidates are bounded in count, time, memory, output, and processes; their "
            "results are scored, never executed further, and a refusal or a candidate failure "
            "is recorded rather than retried around the sandbox"
        ),
    ),
    "eval_contamination": _declared(
        "measure n-gram overlap between evaluation items and a training build",
        people=(
            "compares local evaluation items with a verified tokenized build and records "
            "aggregate overlap rates and per-item coverage, not training text; it runs no "
            f"model and publishes nothing; {_LOCAL_AND_OFFLINE}"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; benchmark files must be registered for evaluation, and a "
            "finding is a lower bound reported for human review, never a clearance"
        ),
        system=(
            "read-only over the build and items, bounded n-gram settings, a content-addressed "
            "report written once, and cooperative cancellation"
        ),
    ),
    "report_review": _declared(
        "read, list, and compare stored evaluation and scaling reports",
        people=(
            f"reads local content-addressed reports and the shipped ladder; {_LOCAL_AND_OFFLINE}"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; every report is verified against its content address before "
            "it is shown, and comparisons warn when reports are not comparable"
        ),
        system="read-only and side-effect free apart from printed output",
    ),
    "scale_profile": _declared(
        "measure training-step throughput and memory for one local configuration",
        people=(
            "trains a freshly initialized local model on seeded synthetic tokens for a bounded "
            "number of steps and records only timings and memory; no data, model, or text "
            f"leaves the machine; {_LOCAL_AND_OFFLINE}"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; only configurations within the local kernel bound are "
            "allocated, and MFU stays unknown unless the operator supplies a device peak"
        ),
        system=(
            "local CPU only with no paid compute, at most 1,000 profiled steps, thread and "
            "determinism settings restored afterwards, and cooperative cancellation between steps"
        ),
    ),
    "scale_sweep": _declared(
        "train a bounded model-size by token-budget by seed sweep on one tokenized build",
        people=(
            "trains local proxy-scale models of at most 50M parameters on a verified, "
            "reviewed tokenized build exactly as kernel_train does; rungs above the bound are "
            "recorded as projection-only and never allocated; nothing is released or "
            f"served; {_LOCAL_AND_OFFLINE}"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; every point is bounded by the kernel policy, failed points are "
            "kept in the report, and losses are labelled proxy-scale measurements"
        ),
        system=(
            "local CPU only with no paid compute, one checkpoint kept per point, resumable "
            "points, and cooperative cancellation that is never gated"
        ),
    ),
    "scale_mixture": _declared(
        "train one small model per domain mixture and measure per-domain validation loss",
        people=(
            "selects deterministic subsets of an already reviewed tokenized build's training "
            "split and trains local proxy-scale models of at most 50M parameters; no build, "
            "tokenizer, or shard is rewritten and nothing is released or served; "
            f"{_LOCAL_AND_OFFLINE}"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; mixtures are matched on training tokens and the report "
            "decides nothing on its own: a human chooses a mixture from measured losses"
        ),
        system=(
            "local CPU only with no paid compute, one checkpoint kept per mixture, resumable "
            "points, and cooperative cancellation that is never gated"
        ),
    ),
    "scale_fit": _declared(
        "fit scaling-law forms to a stored sweep report",
        people=f"arithmetic over a verified local report; {_LOCAL_AND_OFFLINE}",
        direction=(
            f"{_OPERATOR_REQUEST}; applying the published forms to EPOR is labelled a "
            "hypothesis and small-span or under-determined fits carry warnings"
        ),
        system="deterministic, bounded arithmetic and a content-addressed report written once",
    ),
    "scale_project": _declared(
        "project compute, storage, energy, and cost for a target-size run",
        people=(
            "arithmetic over operator-written inputs and a model config; it never fetches a "
            "price or hardware number, never provisions compute, and never calls a paid API; "
            f"{_LOCAL_AND_OFFLINE}"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; a projection is not an approval: every target run still "
            "requires a named human's explicit approval of the exact immutable proposal (G5), "
            "and every quantity with a missing input is reported as unknown"
        ),
        system="read-only over its inputs and a content-addressed report written once",
    ),
    "align_register": _declared(
        "register operator-reviewed teacher outputs and training sources, and seed fixtures",
        people=(
            "reads only local files the operator holds rights to, or the project's own authored "
            "fixtures; refuses a source whose data-engine registration does not allow training, "
            "a teacher whose licence does not permit training on its outputs, and a paid source "
            "without a cost-approval record; it calls no model API, downloads nothing, loads no "
            "external weights, and writes provenance records only under the private data root; "
            f"{_LOCAL_AND_OFFLINE}"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; available only from the owner's CLI, and file contents are "
            "always data, never an instruction"
        ),
        system=(
            "immutable content-addressed records that keep full lineage, so a later tombstone "
            "removes every derived record"
        ),
    ),
    "align_generate": _declared(
        "sample teacher outputs, critique/revision candidates, and model-written G10 items",
        people=(
            "runs hash-verified local proxy-scale checkpoints of at most 50M parameters whose "
            "text has no informational content; every output is stored as a candidate record "
            "under the private data root, is never shown as advice, served, or published, and "
            "reaches training only through an AI-judged run a human has audited. Critique cites "
            "covenant obligations by key from the covenant file. This rationale does not survive "
            "a capable model: one able to produce harmful text needs output classification under "
            f"no-harmful-output before this declaration can be reused; {_LOCAL_AND_OFFLINE}"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; prompts come from project fixtures or records the operator "
            "names, and model text is always data, never an instruction"
        ),
        system=(
            "bounded counts and reply lengths, seeded sampling restricted to learned token IDs, "
            "immutable records, and cooperative cancellation"
        ),
    ),
    "align_judge": _declared(
        "record AI judge decisions over candidate records with a local EPOR checkpoint",
        people=(
            "decisions are stored as data with the judge's identity and lineage and admit "
            "nothing to training on their own: a run's candidates train only after a human "
            "audits a sample of its accepted and rejected decisions and the judge-human "
            f"disagreement rate is within its threshold; {_LOCAL_AND_OFFLINE}"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; the judge cites covenant obligations by key and never "
            "marks anything audited"
        ),
        system="read-only over checkpoints, immutable records, and cooperative cancellation",
    ),
    "align_dataset": _declared(
        "build a content-addressed SFT or preference dataset from provenance records",
        people=(
            "excludes every record a tombstone removes, directly or through its lineage, "
            "refuses records without training-on-outputs rights or whose source no longer "
            "allows training, admits model-authored candidates only from audited judging runs, "
            f"and records every exclusion; it publishes nothing; {_LOCAL_AND_OFFLINE}"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; the operator names the records and the admission rule is "
            "fixed by the audit ledger, not by the request"
        ),
        system="deterministic, immutable, content-addressed output written once",
    ),
    "align_sft": _declared(
        "fine-tune or distill a bounded dense model on a verified chat dataset",
        people=(
            "trains within the kernel bounds from a pinned local checkpoint on a verified "
            "dataset that refuses removed records; logit distillation reads only pinned EPOR "
            "checkpoints sharing the student's tokenizer, and no external weights load into or "
            "seed an EPOR checkpoint; the result is a local proxy-scale research artifact that "
            "is neither released nor served, and its only behaviour claims are G10 comparisons "
            f"that fail closed or escalate to a human; {_LOCAL_AND_OFFLINE}"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; every setting is typed and bounded by the kernel policy and "
            "conversation text is only ever embedded, never interpreted as an instruction"
        ),
        system=(
            "local CPU only with no paid compute, loopback-only ranks, hashed atomic "
            "checkpoints, exact resume, and cooperative cancellation that is never gated"
        ),
    ),
    "align_preference": _declared(
        "optimize a bounded policy on preference pairs against a frozen reference (DPO)",
        people=(
            "starts from a pinned local checkpoint and reads a pinned reference it never "
            "writes; AI-judged pairs enter only after human audit; a per-step log-ratio "
            "monitor stops the run at a checkpoint when its hypothesis abort threshold is "
            "crossed; the result is a local proxy-scale research artifact, neither released nor "
            f"served, and is not evidence of alignment; {_LOCAL_AND_OFFLINE}"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; settings are typed and bounded, and resume refuses any "
            "change to the trajectory"
        ),
        system=(
            "one local CPU process with no paid compute, hashed atomic checkpoints, exact "
            "resume, and cooperative cancellation that is never gated"
        ),
    ),
    "align_audit": _declared(
        "queue AI judge decisions for human audit and record human verdicts",
        people=(
            "lets a verified human review candidate text locally and reject what should not "
            "train; nothing is ever marked audited except by a human verdict, and a rejection "
            f"keeps its candidate out of every dataset; {_LOCAL_AND_OFFLINE}"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; verdicts are human direction recorded with the verified "
            "actor, under a scope separate from training so neither implies the other"
        ),
        system=(
            "an append-only hash-chained log under the private safety root in which no "
            "verdict is ever deleted, only superseded"
        ),
    ),
    "align_evaluate": _declared(
        "measure overoptimization, judge calibration, and model-level G10 behaviour",
        people=(
            "reads local checkpoints, records, and human verdicts and stores content-addressed "
            "measurements; G10 regressions fail closed and remaining failures are queued for "
            "human review, and no number is presented as evidence of safety or alignment; "
            f"{_LOCAL_AND_OFFLINE}"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; every threshold is labelled a hypothesis and a missing "
            "audit is reported as unknown rather than filled in by the judge"
        ),
        system="read-only over its inputs, bounded generation, and cooperative cancellation",
    ),
    "context_extend": _declared(
        "continue a bounded checkpoint at a longer context or with a sparse-attention retrofit",
        people=(
            "continues local proxy-scale training of a hash-verified checkpoint of at most 50M "
            "parameters on a verified, reviewed tokenized build, changing only the context "
            "length, its RoPE scaling, or an experimental sparse-attention indexer; training "
            "at a length never raises the validated context, the result is neither released "
            f"nor served, and nothing is published; {_LOCAL_AND_OFFLINE}"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; any other change to the model is refused by name, every "
            "setting is bounded by the kernel policy, and token IDs are never instructions"
        ),
        system=(
            "local CPU only with no paid compute, loopback-only ranks, hashed atomic "
            "checkpoints, exact resume, and cooperative cancellation that is never gated"
        ),
    ),
    "context_evaluate": _declared(
        "score a checkpoint on the locally generated G6 long-context suite",
        people=(
            "scores a hash-verified local checkpoint on project-authored, benign generated "
            "items and one tokenized split; prompt-injection items only measure whether the "
            "model prefers a planted fact, nothing is downloaded or published, and no score "
            f"is presented as a capability claim; {_LOCAL_AND_OFFLINE}"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; item text, including embedded instructions, is data scored "
            "by likelihood and never acted on"
        ),
        system="read-only over the checkpoint and build, a content-addressed report written once",
    ),
    "context_profile": _declared(
        "measure TTFT, decode throughput, peak RSS, and out-of-memory behaviour per length",
        people=(f"measures local models on seeded synthetic tokens only; {_LOCAL_AND_OFFLINE}"),
        direction=(
            f"{_OPERATOR_REQUEST}; lengths whose certain allocations exceed the memory budget "
            "are refused before running, and the report states that timings describe this "
            "host only"
        ),
        system=(
            "every measurement runs in its own spawned subprocess whose address space is "
            "capped at start-up size plus the budget and which is killed at a wall-clock "
            "timeout, so an out-of-memory or slow length fails cleanly instead of exhausting "
            "or hanging the host"
        ),
    ),
    "context_sweep": _declared(
        "sweep window, local:global ratio, or split RoPE base against full attention",
        people=(
            "trains local proxy-scale models of at most 50M parameters on a verified, reviewed "
            "tokenized build exactly as kernel_train does and scores them on generated items; "
            f"nothing is released or served; {_LOCAL_AND_OFFLINE}"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; the report decides nothing: a human chooses from measured "
            "KV bytes, retrieval, and loss"
        ),
        system=(
            "local CPU only with no paid compute, one checkpoint per point, reused finished "
            "points, and cooperative cancellation that is never gated"
        ),
    ),
    "context_project": _declared(
        "project KV cache, FLOPs, time, and cost for the 32K/128K/256K context recipe",
        people=(
            "arithmetic over a model config and operator-written inputs; it never fetches a "
            "price or hardware number, provisions compute, or calls a paid API; "
            f"{_LOCAL_AND_OFFLINE}"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; a projection is not an approval (G5) and every quantity "
            "with a missing input is reported as unknown"
        ),
        system="read-only over its inputs and a content-addressed report written once",
    ),
    "context_certify": _declared(
        "record a validated context length for one checkpoint from passing G6 evidence",
        people=(
            "raises a checkpoint's validated context only when a G6 report of that exact "
            "checkpoint passes the gate at a length within its trained context, with no "
            "short-context regression against a pre-extension baseline and a host profile "
            "that completed at the length; a refusal states every failing reason, and no "
            f"config or checkpoint is edited; {_LOCAL_AND_OFFLINE}"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; available only from the owner's CLI, under a scope "
            "separate from training and evaluation, because a validated context is what "
            "later releases rely on"
        ),
        system="an immutable, content-addressed certificate that cites every report it rests on",
    ),
    "owner_bootstrap": _declared(
        "mint the single local owner credential on first run",
        people=(
            "writes one random secret to a private file owned by the operator "
            "running the command; it reaches no person and no network"
        ),
        direction=(
            "run from the shell of the machine owner, which is the only place a "
            "first credential can legitimately originate; it refuses to run twice, "
            "so it cannot be used to displace an existing owner"
        ),
        system=(
            "private 0600 credential file beneath a 0700 directory, recorded in "
            "hash-chained safety truth as a digest and never in clear"
        ),
    ),
}

# One command-to-action registry behind the CLI, so a new executing command
# without a declared covenant standing fails closed rather than running
# ungoverned.  Help, version, and the roadmap placeholders execute nothing and
# are deliberately absent.
CLI_ACTIONS: dict[str, str] = {
    "doctor": "system_probe",
    "config validate": "config_validate",
    "research sync": "research_sync",
    "research verify": "research_verify",
    "research index": "research_index",
    "data ingest": "data_ingest",
    "data build": "data_build",
    "data remove": "data_remove",
    "data audit": "data_audit",
    "tokenizer train": "tokenizer_train",
    "tokenizer shard": "tokenizer_shard",
    "tokenizer evaluate": "tokenizer_evaluate",
    "train pretrain": "tiny_train",
    "train resume": "tiny_train",
    "train kernel": "kernel_train",
    "train kernel-resume": "kernel_train",
    "eval run": "tiny_eval",
    "eval kernel": "kernel_evaluate",
    "eval suite": "eval_suite",
    "eval contamination": "eval_contamination",
    "eval compare": "report_review",
    "eval reports": "report_review",
    "scale ladder": "report_review",
    "scale reports": "report_review",
    "scale profile": "scale_profile",
    "scale sweep": "scale_sweep",
    "scale mixture": "scale_mixture",
    "scale fit": "scale_fit",
    "scale project": "scale_project",
    "generate": "generate",
    "train sft": "align_sft",
    "train preference": "align_preference",
    "align fixtures": "align_register",
    "align teacher register": "align_register",
    "align teacher import": "align_register",
    "align source import": "align_register",
    "align teacher sample": "align_generate",
    "align critique": "align_generate",
    "align eval-items": "align_generate",
    "align judge": "align_judge",
    "align dataset": "align_dataset",
    "align audit sample": "align_audit",
    "align audit verdict": "align_audit",
    "align audit list": "report_review",
    "align audit calibration": "align_evaluate",
    "align monitor": "align_evaluate",
    "align g10": "align_evaluate",
    "align reports": "report_review",
    "context extend": "context_extend",
    "context retrofit": "context_extend",
    "context g6": "context_evaluate",
    "context profile": "context_profile",
    "context feasibility": "context_profile",
    "context sweep": "context_sweep",
    "context project": "context_project",
    "context certify": "context_certify",
    "context status": "report_review",
    "context reports": "report_review",
    "api": "control_api_start",
    "worker": "job_worker_start",
    "ui": "local_ui_start",
    "auth bootstrap": "owner_bootstrap",
}

UNDECLARED = ProposedAction(
    summary="execute work that has no reviewed covenant declaration",
    assessments=tuple(
        PrincipleAssessment(
            priority=priority,
            status="uncertain",
            confidence=0.0,
            rationale="no reviewed declaration states how this work stands under the covenant",
            obligation=obligation,
        )
        for priority, obligation in zip(
            PRIORITIES,
            ("no-direct-harm", "verify-authorization", "fail-safely"),
            strict=True,
        )
    ),
)


class ActorEvidence(Protocol):
    """What the covenant needs to know about who is asking.

    Structural rather than imported, so the action registry stays free of the
    control plane; :class:`epor.control.authority.Actor` satisfies it.
    """

    @property
    def id(self) -> str: ...

    @property
    def is_authenticated(self) -> bool: ...

    def may(self, action_id: str) -> bool: ...


def _with_actor(action: ProposedAction, action_id: str, actor: ActorEvidence) -> ProposedAction:
    """Fold identity and scope into the second principle's assessment.

    Authorization is not a separate check bolted on after resolution. "Follow
    legitimate human direction" is precisely the principle an unverified or
    out-of-scope request engages, so the evidence belongs inside that
    assessment, where the resolver's ordering can act on it: a Principle 2
    conflict with nothing higher-ranked to justify it refuses, and it can never
    be talked past by a Principle 3 argument about availability or sunk work.
    """

    declared = action.assessment(2)
    if actor.may(action_id):
        replacement = PrincipleAssessment(
            priority=2,
            status=declared.status,
            confidence=declared.confidence,
            obligation=declared.obligation,
            rationale=f"{declared.rationale}; requested by verified actor {actor.id}",
        )
    elif not actor.is_authenticated:
        replacement = PrincipleAssessment(
            priority=2,
            status="conflicted",
            confidence=0.99,
            obligation="verify-authorization",
            rationale="the requester's identity is unverified, so its instruction is not direction",
        )
    else:
        replacement = PrincipleAssessment(
            priority=2,
            status="conflicted",
            confidence=0.99,
            obligation="refuse-unlawful-or-out-of-scope",
            rationale=(
                f"actor {actor.id} holds no delegation covering {action_id!r}, "
                "so the request lies outside its authorized scope"
            ),
        )
    assessments = tuple(replacement if item.priority == 2 else item for item in action.assessments)
    return action.model_copy(update={"assessments": assessments})


def resolve_action(
    action_id: str,
    covenant: Covenant,
    *,
    actor: ActorEvidence | None = None,
) -> Resolution:
    """Resolve whether ``action_id`` may execute, for ``actor`` if supplied.

    An unrecognized or undeclared action is assessed as uncertain against every
    principle, so it escalates to a human rather than running.  Omitting the
    actor resolves the declaration on its own terms; callers that have an
    identity must pass it, because a declaration alone says nothing about
    whether this requester was authorized to invoke it.
    """

    action = DECLARED_ACTIONS.get(action_id, UNDECLARED)
    if actor is not None:
        action = _with_actor(action, action_id, actor)
    return resolve(action, covenant)


def authorization_denied(resolution: Resolution) -> bool:
    """Whether a refusal was decided by scope rather than by the work itself.

    Used only to pick the reported error code.  The refusal itself was already
    decided by the covenant; this reads which obligation bound it.
    """

    return (
        resolution.outcome == "refuse"
        and resolution.binding_priority == 2
        and resolution.binding_obligation
        in {"refuse-unlawful-or-out-of-scope", "verify-authorization"}
    )
