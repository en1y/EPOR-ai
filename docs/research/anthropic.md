# Anthropic research dossier

Source IDs below refer to [`research/catalog.yaml`](../../research/catalog.yaml).
The notes distinguish historical publication, reported evidence, EPOR choices,
and information Anthropic has not made public.

## Historical model and data disclosure

- **`[disclosed]`** `anthropic-general-assistant` describes a historical
  alignment laboratory built around a language assistant and gives a detailed
  approximately 400-billion-token training setting for that research system.
- **`[unknown]`** That 2021 setting is not evidence for the corpus, token count,
  architecture, optimizer, infrastructure, or alignment recipe of a current
  Claude model.
- **`[EPOR-adaptation]`** EPOR records historical recipes as experimental inputs
  and independently chooses data, tokenizer, optimizer, architecture, and scale
  through proxy ablations. No Claude weights are reused.

## Human and AI feedback

- **`[disclosed]`** `anthropic-hh-rlhf` describes helpfulness/harmlessness
  preference collection and an RLHF pipeline using human comparisons.
- **`[reported-result]`** The HH-RLHF paper reports trade-offs between helpful
  and harmless behavior under its models, prompts, annotators, and evaluation
  conditions; those measurements are not EPOR baselines.
- **`[disclosed]`** `anthropic-constitutional-ai` describes supervised
  critique/revision followed by preference learning where a written constitution
  supplies principles and a model supplies feedback.
- **`[EPOR-adaptation]`** EPOR will version its own constitution, retain the
  provenance of every principle and teacher judgment, audit accepted and
  rejected examples with humans, and start with DPO before considering broader
  reinforcement learning.
- **`[hypothesis]`** Constitutional critique/revision plus audited AI preferences
  may reduce harmful responses with less exposure of human raters to disturbing
  material; EPOR must test this against human-preference and SFT controls.

## Repeated data

- **`[reported-result]`** `anthropic-repeated-data` reports a double-descent-like
  degradation when a small subset of training data is repeated at particular
  frequencies and reports damage to copying/generalization structures such as
  induction heads in its experimental models.
- **`[hypothesis]`** The paper proposes that memorizing the repeated subset can
  consume model capacity and help explain the degradation; it does not establish
  that mechanism for every architecture or data regime.
- **`[EPOR-adaptation]`** EPOR measures exact and near-duplicate rates, effective
  unique tokens, per-source repetition, memorization, and validation loss in
  mixture ablations. Deliberate up-weighting is recorded rather than hidden by
  nominal token counts.

## Long context and retrieval

- **`[reported-result]`** `anthropic-long-context-prompting` reports a controlled
  Claude Instant/Claude 2 recall study using synthetic long documents and finds
  that extracting relevant passages and adding task-relevant examples can help
  under the tested prompts; it also reports sensitivity to passage position.
- **`[EPOR-adaptation]`** EPOR context certification separately evaluates
  beginning, middle, and end retrieval, multi-document QA, long code, perplexity
  by position, short-context regression, prompt injection, TTFT, throughput,
  memory, and OOM behavior.
- **`[disclosed]`** `anthropic-contextual-retrieval` prepends chunk-specific
  context before both embedding and lexical indexing, producing contextual
  embedding and contextual BM25 variants.
- **`[reported-result]`** Anthropic reports lower retrieval failure for its tested
  contextual-retrieval configurations, particularly when contextual embedding
  and BM25 signals are combined; EPOR has not reproduced those figures.
- **`[EPOR-adaptation]`** Retrieval augmentation remains an external system
  capability. It does not raise a model's trained or validated context length and
  must be evaluated for provenance, stale evidence, citation quality, and prompt
  injection.

## Claude model card boundary

- **`[disclosed]`** `anthropic-claude-3-model-card` documents intended behavior,
  evaluations, safety work, and selected limitations for the Claude 3 family.
- **`[unknown]`** The card does not publish enough information to reconstruct
  Claude 3's architecture, corpus, tokenizer, full optimizer/training schedule,
  data ordering, or internal distributed trainer.
- **`[EPOR-adaptation]`** EPOR model and system cards explicitly separate what was
  configured, trained, validated, and certified on local hardware, and include
  immutable run/data/tokenizer/code lineage.

## Model-written evaluations

- **`[disclosed]`** `anthropic-model-written-evals` uses language models to
  generate behavioral evaluation questions and applies them to look for model
  tendencies, including behaviors not adequately represented by conventional
  benchmarks.
- **`[reported-result]`** The paper reports that generated evaluations surfaced
  behaviors and scaling trends in its tested model families; generator and judge
  bias remain part of the measurement.
- **`[EPOR-adaptation]`** EPOR may use model-authored candidates, but evaluation
  ownership stays human: specifications, held-out seeds, judge calibration,
  disagreement samples, contamination checks, and manual audits are mandatory.

## Sparse autoencoders and circuit tracing

- **`[disclosed]`** `anthropic-towards-monosemanticity` and
  `anthropic-scaling-monosemanticity` apply dictionary learning/sparse
  autoencoders to decompose activations into learned features, including work on
  Claude 3 Sonnet at larger feature scale.
- **`[reported-result]`** Those sources report features associated with
  recognizable concepts and interventions that alter some related behavior;
  feature descriptions remain imperfect and selective.
- **`[disclosed]`** `anthropic-circuit-tracing` constructs attribution graphs
  intended to expose part of the feature-to-feature computation behind selected
  outputs.
- **`[unknown]`** A sparse feature or attribution graph is not a complete account
  of model computation, a proof of faithfulness, or a safety/correctness
  certificate.
- **`[EPOR-adaptation]`** Later EPOR releases add activation hooks, probes, sparse
  autoencoders, graph-based hypotheses, and causal interventions only after
  proxy-scale validation. Reports include reconstruction error, feature
  stability, intervention specificity, and negative findings.

## Open questions

- **`[unknown]`** Public Anthropic material does not reveal a reproducible modern
  Claude pretraining recipe or internal virtual environment.
- **`[hypothesis]`** Model-written evaluations, mechanistic features, and
  constitutional feedback may be most useful as mutually checking evidence
  streams rather than any one source acting as an oracle.
- **`[EPOR-adaptation]`** EPOR treats these methods as experiments with explicit
  controls and limitations, not as a claim to reproduce Claude.
