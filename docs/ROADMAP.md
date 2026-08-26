# EPOR AI roadmap

This roadmap is a sequence of evidence gates, not a promise that target-scale
training has been funded. Every released EPOR weight starts from an independent
random initialization. Licensed teacher logits, synthetic examples, filtering
judgments, and verified reasoning traces may inform training; external weights
must never seed or be merged into an EPOR checkpoint.

Public names are always **EPOR-α**, **EPOR-β**, and **EPOR-γ**. Machine-safe IDs
are `epor-alpha`, `epor-beta`, and `epor-gamma`. Families are always listed
alphabetically, and intended capability descends in the same order: α is the
flagship, γ the compact local model. Training order is the reverse, because the
gated program promotes the smallest design first.

| Model      | Purpose                                            | Architecture destination                                                                                                                                                               | Configured context target | Local v1 certification goal |
|------------|----------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------:|----------------------------:|
| **EPOR-α** | Highest-capability reasoning, code, and math model | About 30B total MoE parameters, 6–8B measured active per token, shared plus fine-grained routed experts, bias-based load balancing, and bounded routing fan-out                        |                      256K |                          8K |
| **EPOR-β** | Balanced general, code, and reasoning model        | 10–12B dense decoder with RMSNorm, SwiGLU, GQA, RoPE, QK-norm, and interleaved bounded-window local/global attention                                                                   |                      256K |                         16K |
| **EPOR-γ** | Compact, knowledge-efficient local model           | About 8B total parameters; nested ≈4B accelerator-resident core; optional ≈2B slice; MatFormer-style elasticity, multi-teacher distillation, and gated PLE/conditional-memory research |                      128K |                         32K |

Every architectural mechanism named in this roadmap is a candidate with a
deciding gate, not a settled choice. The [technique adoption
ledger](#technique-adoption-ledger) lists each one, its evidence label, and the
gate that decides it. A mechanism published by another lab is evidence that it
worked in that lab's setting, never evidence that EPOR should ship it.

## Project safety covenant

Every roadmap version must uphold the following project principles, inspired
by Isaac Asimov's Three Laws of Robotics and applied in strict priority order:

1. **Protect people.** EPOR must not harm a person or knowingly allow
   preventable harm through inaction.
2. **Follow legitimate human direction.** EPOR must follow authorized human
   instructions unless doing so conflicts with the first principle.
3. **Preserve the system responsibly.** EPOR may protect its operation,
   integrity, and availability only when that does not conflict with the first
   two principles.

These principles govern data acquisition, model behavior, training rewards,
tool use, access control, refusal and escalation, shutdown behavior, and release
decisions. They must be backed by measurable evaluations, defense in depth,
human oversight, and documented limitations; the fictional laws alone are not
a complete safety system.

The authoritative form of the covenant is `configs/covenant-v1.yaml`: versioned,
hashable, and executed by the ordered resolver in `epor.safety`. The prose above
is a reading of that file, not a second source of truth. The resolver enforces
ordering over supplied assessments; it does not produce them and is not a harm
detector. Building the assessments, the authorization model, and the evaluations
is the work of v0.0.2 onward.

## v0.0.1 — research-backed bootstrap

### Repository and research foundation

- Write `.gitignore` before downloading anything. Ignore virtual environments,
  raw and extracted research, datasets, checkpoints, runs, weights, GGUF files,
  UI build output, logs, caches, and SQLite runtime state.
- Provide a root README, documentation index, this roadmap, architecture
  overview, data-governance policy, compute policy, and contribution/security
  guidance.
- License project code under Apache-2.0. Assign dataset and weight licenses only
  after their own provenance audits.
- Track `research/catalog.yaml` as the source ledger. Keep `research/raw/`,
  `research/extracted/`, and `research/cache/` ignored. Track only original,
  paraphrased dossiers under `docs/research/`.
- Record canonical URL, title, authors, organization, publication or version
  date, retrieval time, media type, SHA-256 state, license/access notes, tags,
  and ignored local paths for every source.
- Implement `epor research sync|verify|index` with exact-host URL allowlisting,
  atomic streaming downloads, content-size/type checks, checksum verification,
  idempotence, offline operation, and local text extraction.
- Classify every research assertion as `disclosed`, `reported-result`,
  `EPOR-adaptation`, `hypothesis`, or `unknown`.

The initial archive covers:

- Anthropic's historical model/data disclosure, HH-RLHF, Constitutional
  AI/RLAIF, repeated-data effects, long-context behavior, contextual retrieval,
  Claude model cards, model-written evaluations, sparse autoencoders, and
  circuit tracing. **`[disclosed]`** The 400B-token recipe in [A General Language Assistant as a
  Laboratory for Alignment](https://arxiv.org/abs/2112.00861) is historical
  research. **`[unknown]`** It is not a current Claude recipe, and modern architecture and data details
  remain undisclosed in the [Claude 3 model
  card](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model-Card-Claude-3.pdf).
  Alignment adaptations draw from [HH-RLHF](https://arxiv.org/abs/2204.05862)
  and [Constitutional AI](https://arxiv.org/abs/2212.08073).
- DeepSeek LLM, MoE, Coder, Math, V2, Coder-V2, V3, R1, V3.2, Engram, and
  official infrastructure repositories. Anchors include [DeepSeek
  LLM](https://arxiv.org/abs/2401.02954),
  [DeepSeek-V2](https://arxiv.org/abs/2405.04434),
  [DeepSeek-V3](https://arxiv.org/abs/2412.19437),
  [DeepSeek-R1](https://arxiv.org/abs/2501.12948), and
  [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556).
- Gemma 3, Gemma 3n, and MatFormer. **`[disclosed]`** Google describes Gemma 3n E4B as roughly
  8B raw parameters with about 4B effective/core Transformer parameters kept in
  accelerator memory through MatFormer and Per-Layer Embedding techniques. **`[EPOR-adaptation]`** That
  disclosure motivates EPOR-γ; it does not demonstrate that EPOR can reproduce
  Google's recipe. See the [Gemma 3n
  overview](https://ai.google.dev/gemma/docs/gemma-3n) and [developer
  guide](https://developers.googleblog.com/introducing-gemma-3n-developer-guide/).

### Reproducible environments

- Pin Python `3.11.x` in `.python-version`; use `uv`, commit `uv.lock`, use
  frozen synchronization, and document `uv run` as the execution path.
- Keep the default environment CPU-safe. Use explicit optional capability
  profiles: `dev`, `research`, `data`, `train-cpu`, `eval`, `serve`, and `ui`.
  Training-provider profiles are mutually exclusive; a future provider-pinned
  `train-cuNNN` profile will conflict with `train-cpu`.
- Core/control packages: Pydantic, Typer, Rich, PyYAML, HTTPX, FastAPI,
  Uvicorn, SQLAlchemy 2, Alembic, and psutil.
- Research/data packages: pypdf, Hugging Face Datasets and Tokenizers, PyArrow,
  Polars, DuckDB, fsspec, WARC tooling, MinHash/dedup utilities, and xxhash.
- Training packages: PyTorch, Safetensors, Transformers interoperability,
  Accelerate, einops, and TensorBoard.
- Later scale packages remain optional: DeepSpeed, TRL, Ray, Triton,
  FlashAttention, vLLM, SGLang, DeepGEMM, FlashMLA, and DeepEP.
- Quantization uses torchao, whose `QATConfig` prepare/convert flow inserts fake
  quantization for training and then swaps in real quantized ops, including
  int4 weight configurations. It is PyTorch-native, so it fits the pure-PyTorch
  contract without a second trainer. It enters as an optional profile at v0.2.0
  when γ's QAT gate opens, and the GGUF converter remains the authority on what
  actually ships: a torchao result that llama.cpp cannot reproduce within the
  published drift budget is not a release artifact.
- Evaluation packages: lm-eval, SymPy, tree-sitter, and sandboxed compiler/test
  adapters. Development packages: pytest, pytest-cov, Hypothesis, Ruff, and
  Pyright.
- Keep JAX and Hopper-specific kernels in separate environments. **`[disclosed]`** Anthropic has
  disclosed use of PyTorch, JAX, and Triton, but does not publish its internal
  environment or dependency pins.
- Use PyTorch's composable [`fully_shard`/FSDP2
  path](https://github.com/pytorch/pytorch/blob/main/docs/source/distributed.fsdp.fully_shard.md)
  and distributed checkpointing for future scale-out while retaining a
  single-process reference loop.
- Treat the RX 5700 XT as a llama.cpp/Vulkan inference target, not a required
  PyTorch training accelerator. **`[disclosed]`** AMD's current [ROCm
  compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
  does not list the RX 5000 family among validated Radeon targets.

### Runnable reference system

- Build a pure-PyTorch 10–50M parameter decoder-only model with RMSNorm,
  SwiGLU, GQA, RoPE, causal masking, tied embeddings, deterministic
  initialization, and CPU-safe FP32.
- Supply a generated fixture corpus and debug tokenizer so CI never downloads
  data.
- Provide safe commands for config validation, tiny pretraining, evaluation,
  generation, checkpoint saving, and exact resume.
- Provide target-family configurations that instantiate on the meta device for
  parameter counting without allocating 8B–30B weights.
- Every model configuration exposes display name, ASCII slug, total and
  active/core parameter estimates, `configured_max_context`,
  `trained_max_context`, `validated_max_context`,
  `operational_default_context`, and hardware-specific certified profiles.
- Every run manifest records code revision, config hash, data/tokenizer hashes,
  seeds, dependencies, hardware, precision, timestamps, checkpoints, metrics,
  and parent run.

### Local management console

- Implement a loopback-only FastAPI control plane, separate worker, SQLite in
  WAL mode, SQLAlchemy 2 typed models, Alembic migrations, and a React
  19/TypeScript frontend.
- v0.0.1 allows only `system_probe`, `research_sync`, `tiny_train`, and
  `tiny_eval`. Client-supplied shell commands are forbidden.
- Implement transitions `queued → starting → running → succeeded|failed`,
  `queued → cancelled`, and `starting|running → cancelling →
  cancelled|failed`. Mark stale heartbeats `interrupted`; retry creates a new,
  linked job.
- Initial pages show system capabilities/environment health, research catalog
  status, model-family plan/configuration cards without invented metrics, and
  job progress, logs, cancellation, checkpoints, and artifact checksums.
- Bind to `127.0.0.1`, enforce exact-origin CORS, redact secrets, contain paths
  below configured roots, and expose neither arbitrary URL fetches nor a general
  filesystem browser.
- Poll append-only events in v0.0.1. SSE may be added later without changing
  stored event semantics.
- Treat SQLite as a rebuildable index; immutable file manifests remain the
  source of truth.

## Version ladder

### v0.0.x — prove every subsystem cheaply

- **v0.0.1 — Research-backed bootstrap:** Complete the repository, ignored paper
  archive, research dossiers, locked environment, reference model,
  deterministic tiny training, manifests, safe worker, and local console above.
- **v0.0.2 — Safety covenant foundation:** Land the covenant as executable
  contract before the data engine exists, because every later version inherits
  the decisions it governs. In place: a versioned, hashable, machine-readable
  covenant (`configs/covenant-v1.yaml`) stating each principle as an operative
  law with precisely worded obligations; the ordered resolution kernel
  (`epor.safety`), which permits a conflict only when the action protects a
  strictly higher-ranked principle, requires every claimed conflict or doubt to
  cite an obligation by key, escalates when one principle stands on both sides,
  and escalates rather than proceeds under uncertainty; a single action
  registry (`epor.actions`) shared by every entry point, so nothing is governed
  on one path and ungoverned on another; enforcement at control-plane admission
  before a job spec is even parsed, with the resolution and covenant hash
  written into append-only file truth, a worker re-check at dispatch, and the
  same gate on every executing CLI command; and the opening G10 suite. An
  action with no reviewed declaration escalates rather than running, so adding
  executable work requires stating its covenant standing first. Still to do in
  this version: an authorization model distinguishing owner, delegated
  operator, and unauthenticated caller; a documented escalation path with an
  operator-visible queue; and covenant surfacing in the console. Ratify the
  draft covenant at the end of this version; no later version may weaken a
  principle without a version bump and a recorded rationale.
- **v0.0.3 — Provenance-first data engine:** Add rights-aware registration,
  streaming ingestion, a normalized document schema, exact and global
  near-deduplication, language/domain/quality classification, PII and credential
  removal, malware/unsafe-content quarantine, robots/terms records, removal
  tombstones, content-hash splits, and dataset cards.
- **v0.0.4 — Tokenizer and dataset v0:** Train an original 65,536-ID byte-level
  BPE with 256 reserved IDs for roles, tools, FIM, documents, retrieval, and
  future extensions. Preserve whitespace/code exactly, guarantee byte fallback,
  measure domain fertility, and write content-addressed token shards with
  document boundaries.
- **v0.0.5 — Training kernel:** Add token-based batching, AdamW (`β1=.9`,
  `β2=.95`, weight decay `.1`, clip `1.0`), warmup/cosine scheduling, sequence
  packing, BF16 mixed precision, activation checkpointing, DDP, FSDP2,
  distributed checkpoints, fault injection, and exact data-cursor/RNG resume.
  Add QK-norm to the reference block and keep the pre-norm variant as the
  control. **`[disclosed]`** Multi-token prediction is a training objective in
  [DeepSeek-V3](https://arxiv.org/abs/2412.19437), carrying its own weighted
  loss and a full causal chain per depth, not only an inference-time trick.
  **`[EPOR-adaptation]`** EPOR implements it behind a flag and decides it at
  G3/G5 against single-token training at matched tokens, reporting whether the
  extra heads pay for their memory. **`[disclosed]`** DeepSeek-V3 also
  describes FP8 mixed-precision training with fine-grained quantization.
  **`[EPOR-adaptation]`** BF16 is EPOR's baseline and FP8 stays a cloud-only
  ablation gated on loss-curve agreement with a BF16 control; the reference
  machine cannot run it, so it never becomes the default path.
- **v0.0.6 — Evaluation and scaling laboratory:** Add validation
  loss/perplexity, code/math/general benchmarks, calibration, contamination and
  memorization tests, safety suites, hardware profiling, data-mixture ablations,
  and 10M→100M→300M→1B sweeps. Project tokens, GPU-hours, storage, failure
  reserve, and cost before any target-size run.
- **v0.0.7 — Knowledge transfer and alignment:** Add teacher-data provenance,
  response/logit distillation, SFT, pairwise preferences, a DPO baseline,
  constitutional critique/revision, AI-judged preferences with human audits,
  and reward-overoptimization monitoring. Critique and revision cite the
  covenant ratified in v0.0.2; alignment training must not restate its
  principles in a second, drifting form. Extend G10 from resolver-level
  ordering to model-level behavior under the same six situations.
  **`[disclosed]`** [Constitutional AI](https://arxiv.org/abs/2212.08073) pairs
  supervised critique/revision with preference learning driven by a written
  constitution and model-supplied feedback, and
  [model-written evaluations](https://arxiv.org/abs/2212.09251) generate
  behavioral test items with a model. **`[EPOR-adaptation]`** Both enter as
  candidate generators only: every AI-authored preference and evaluation item
  carries provenance, a human-audited sample of accepted *and* rejected cases,
  judge-calibration data, and a disagreement rate. **`[hypothesis]`** Model
  feedback may reduce human exposure to disturbing material without a quality
  loss; that must beat SFT and human-preference controls before it becomes the
  default. **`[disclosed]`** Gemma 3 and MatFormer both describe distillation
  during training. **`[EPOR-adaptation]`** γ's multi-teacher distillation runs
  here, over licensed teacher outputs only, with the teacher, its license, and
  the exact sampled outputs recorded per example; no external weights ever seed
  or merge into an EPOR checkpoint.
- **v0.0.8 — Context laboratory:** Train progressively at 4/8K→32K→128K and
  then 256K for α/β only after 128K passes. Evaluate RoPE/YaRN, GQA cache
  quantization, MLA, long-document mixtures, lost-middle behavior, prompt
  injection, TTFT, throughput, and memory. **`[disclosed]`** The [Gemma 3
  report](https://arxiv.org/abs/2503.19786) interleaves five bounded-window
  local attention layers per global layer, raises the RoPE base frequency on
  global layers to one million while leaving local layers at ten thousand, and
  extends context by an interpolation-style procedure. **`[reported-result]`**
  It observes little perplexity movement across local-to-global ratios from 1:1
  to 7:1 in its own study. **`[EPOR-adaptation]`** Ratio, window size, and the
  split RoPE base are three separate sweeps here against a full-attention
  control, judged on KV-cache bytes per token and position-wise retrieval rather
  than perplexity alone, because a ratio that is free in perplexity can still
  lose the middle of a long document. **`[disclosed]`**
  [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) introduces sparse attention
  through continued training on an otherwise unchanged architecture, pairing a
  lightweight index score with fine-grained per-query token selection.
  **`[EPOR-adaptation]`** EPOR therefore treats sparse attention as a retrofit
  evaluated against an already-certified dense or local/global checkpoint, so a
  failed experiment delays no release.
- **v0.0.9 — Export, inference, and full console:** Add Hugging
  Face/Safetensors export, GGUF conversion, Q8/Q6/Q5/Q4 profiles, llama.cpp
  Vulkan integration, quantization parity reports, a provider-neutral local
  generation API, chat playground, registries for models/datasets/tokenizers/
  evaluations, a cost estimator, and artifact comparison.
- **v0.0.10 — Three-family proxy release:** Train and publish tiny/proxy
  EPOR-α, EPOR-β, and EPOR-γ checkpoints with separate model cards, scaling
  results, local benchmarks, and architecture-ablation reports.
- **v0.1.0 — First reproducible research preview:** Release stable
  tokenizer/data/config/run-manifest formats, proxy weights, GGUF artifacts,
  documented one-command reproduction, the local console, and a full
  limitations report.

### v0.2–v0.9 — gated target-model program

- **v0.2.0 — EPOR-γ:** Promote the 8B-total/≈4B-core design only after
  MatFormer-style slice training and static extraction work at proxy scale. A
  conventional standalone ≈4B GGUF slice is mandatory even if dynamic PLE
  execution is delayed. Train with multi-teacher distillation and measure
  knowledge retained per active FLOP, GB, and latency. Target 128K cloud
  validation and 32K on the reference PC. **`[disclosed]`** Gemma 3 produces its
  quantized releases by a short quantization-aware fine-tune that targets the
  unquantized checkpoint's own output probabilities. **`[EPOR-adaptation]`** γ
  is the family where local memory decides usefulness, so QAT moves here from
  the later efficiency track and its parity report is a release requirement, not
  an optimization. **`[disclosed]`** The [Engram
  source](https://arxiv.org/abs/2601.07372) frames conditional memory as a
  sparsity-allocation problem between computation and static lookup, reports an
  interior optimum rather than a monotonic preference, and relies on
  deterministic addressing to prefetch from host memory.
  **`[EPOR-adaptation]`** EPOR adopts the allocation framing and sweeps its own
  split against compute-matched dense and MatFormer controls; deterministic
  addressing is a prerequisite, since a lookup that cannot be prefetched defeats
  the accelerator-residency goal that motivates γ at all.
- **v0.3.0 — EPOR-β:** Train the 10–12B dense model with the selected data
  mixture and scaling-law hyperparameters. Target a practical Q4 Vulkan
  artifact, 256K cloud validation, and a 16K local profile.
- **v0.4.0 — EPOR-α:** Train the ≈30B-total MoE with shared and fine-grained
  routed experts. Proxy ablations select top-k routing and balancing policy.
  Require 6–8B measured active parameters, no dead experts, acceptable routing
  communication, GGUF parity, 256K cloud validation, and an 8K local profile.
  **`[disclosed]`** DeepSeek-V3 balances load by adjusting a per-expert routing
  bias at a configured update speed, decayed to zero late in training, keeping
  only a very small sequence-wise balance term against extreme imbalance, and
  bounds communication by capping how many nodes a token may reach.
  **`[EPOR-adaptation]`** Bias-based balancing is decided against an
  auxiliary-loss control at proxy scale on expert utilization, dead-expert count,
  and quality — not adopted because it is newer. **`[unknown]`** The published
  update speed, node cap, and MTP weight schedule come from a cluster run three
  orders of magnitude larger than α; they seed a sweep and are never copied as
  settings. **`[EPOR-adaptation]`** Any balancing or routing policy that cannot
  be expressed in a GGUF export is rejected at this gate regardless of measured
  quality, because an α that only runs in PyTorch fails the local-first premise.
- **v0.5.0 — Reasoning specialization:** Build human-cleaned cold starts,
  verified math/code traces, rejection sampling, compiler/test/SymPy rewards,
  and limited GRPO for α. Distill verified α reasoning into β and γ. Do not use
  unverifiable free-form rewards as the primary RL signal. **`[reported-result]`**
  [DeepSeek-R1](https://arxiv.org/abs/2501.12948) reports that an RL-first
  variant developed readability and language-mixing problems, which is why it
  describes human-readable cold-start data before RL. **`[EPOR-adaptation]`**
  EPOR treats that as the load-bearing finding rather than the headline scores:
  cold starts come first, and readability and language consistency are scored
  gates on every RL checkpoint, not post-hoc observations. **`[disclosed]`**
  GRPO originates in [DeepSeek-Math](https://arxiv.org/abs/2402.03300).
  **`[EPOR-adaptation]`** Its scope stays limited to α with verifiable rewards;
  reward-model-only signals are monitored for overoptimization and never become
  the primary objective.
- **v0.6.0 — Safety and interpretability:** Revise the ratified covenant under
  its own version discipline, held-out red teaming, input/output classifiers,
  calibration and abstention, activation hooks, probes, sparse autoencoders,
  and causal interventions. Treat interpretability as evidence, not a
  correctness certificate.
- **v0.7.0 — Context certification:** Complete position-wise and task-level
  validation for α/β at 256K and γ at 128K on cloud hardware. Certify lower
  operational profiles on the exact Ryzen/RX 5700 XT/32 GiB reference system.
- **v0.8.0 — Advanced efficiency:** Carry forward whatever the earlier gates
  left undecided — MLA against GQA plus cache quantization, sparse attention as
  a retrofit, expert streaming, and Colibri integration — and add speculative
  decoding. **`[reported-result]`** [MatFormer](https://arxiv.org/abs/2310.07707)
  reports speculative-decoding benefits from an extracted submodel that shares
  behavior with the full model. **`[EPOR-adaptation]`** γ's nested slices are
  therefore its own draft models, evaluated on accepted-token rate and
  end-to-end latency rather than on draft accuracy alone; multi-token prediction
  heads from v0.0.5 are the competing draft source and the two are compared
  directly. Colibri remains optional until measured usable. **`[disclosed]`**
  Its official project demonstrates disk-streamed MoE execution with
  architecture-specific runtime work
  ([Colibri](https://github.com/JustVugg/colibri)).
- **v0.9.0 — Public release candidate:** Perform fresh-machine reproduction,
  data/license/privacy review, model and system cards, signed hashes, SBOMs,
  corrupted-artifact handling, soak/OOM/cancellation testing, API migration
  documentation, and external red-team review.
- **v1.0.0 — Stable open family:** Release base and instruct checkpoints for all
  three models where compute gates were funded and passed, stable tokenizer/chat/
  tool schemas, HF and GGUF artifacts, local API/UI, benchmark and hardware
  matrices, reproducibility records, and explicit limitations.

### Beyond v1

- **v1.1:** Continual data refresh, opt-out/removal propagation, regression-safe
  continued pretraining, and weight-delta lineage.
- **v1.2:** Tool-use and agentic task synthesis with sandboxed execution and
  approval boundaries.
- **v1.3:** Carefully measured multilingual expansion.
- **v1.4:** Optional multimodal γ research; text-only remains the v1 contract.
- **v2.0:** Conditional-memory and sparse-compute redesigns, elastic family
  execution, distributed local inference, and experimentally validated context
  beyond 256K, including a possible 1M-token track.

## Technique adoption ledger

Every mechanism EPOR borrows, the evidence behind it, the gate that decides it,
and the control it has to beat. A technique with no deciding gate or no control
is not on the roadmap — it is a preference, and preferences do not enter
checkpoints. Sources are catalog IDs in
[`research/catalog.yaml`](../research/catalog.yaml); dossiers are under
[`docs/research/`](research/).

| Technique                                          | Source                                                           | Evidence                                                 | Decided at                        | Must beat                                                              |
|----------------------------------------------------|------------------------------------------------------------------|----------------------------------------------------------|-----------------------------------|------------------------------------------------------------------------|
| RMSNorm, SwiGLU, GQA, RoPE, tied embeddings        | common practice                                                  | `disclosed`                                              | shipped in v0.0.1 reference model | —                                                                      |
| QK-norm                                            | `gemma-3-technical-report`                                       | `disclosed`                                              | v0.0.5                            | pre-norm block without QK-norm                                         |
| Interleaved local/global attention, bounded window | `gemma-3-technical-report`                                       | `disclosed`, ratio study `reported-result`               | v0.0.8, G6                        | full attention, on KV bytes/token *and* position-wise retrieval        |
| Split RoPE base (local vs global layers)           | `gemma-3-technical-report`                                       | `disclosed`                                              | v0.0.8, G6                        | single base with YaRN extension                                        |
| Multi-token prediction as a training objective     | `deepseek-v3`                                                    | `disclosed`                                              | v0.0.5, G3/G5                     | single-token training at matched tokens                                |
| FP8 mixed precision                                | `deepseek-v3`                                                    | `disclosed`                                              | v0.0.5, cloud only                | BF16 loss-curve agreement; never the reference-machine path            |
| Bias-based (auxiliary-loss-free) load balancing    | `deepseek-v3`                                                    | `disclosed`; published constants `unknown` at EPOR scale | v0.4.0, G7                        | auxiliary-loss balancing on utilization, dead experts, quality         |
| Node-limited routing                               | `deepseek-v3`                                                    | `disclosed`                                              | v0.4.0, G7                        | unrestricted routing on communication cost vs quality                  |
| Shared plus fine-grained routed experts            | `deepseek-moe`                                                   | `disclosed`, trade-offs `reported-result`                | v0.4.0, G7                        | compute-matched dense control                                          |
| Multi-head Latent Attention                        | `deepseek-v2`, `deepseek-v3`                                     | `disclosed`; benefit at EPOR scale `hypothesis`          | v0.0.8, v0.8.0                    | GQA plus cache quantization, including GGUF parity                     |
| DeepSeek Sparse Attention                          | `deepseek-v3-2`                                                  | `disclosed`                                              | v0.0.8 retrofit, v0.8.0           | an already-certified dense or local/global checkpoint                  |
| MatFormer nested slices                            | `matformer`, `gemma-3n-overview`                                 | `disclosed`; slice quality `hypothesis`                  | G4, v0.2.0                        | independently trained compute-matched dense per slice                  |
| PLE / Engram-style conditional memory              | `gemma-3n-overview`, `deepseek-engram`                           | `disclosed`; usefulness `hypothesis`                     | G4, v0.2.0                        | dense and MatFormer controls on knowledge per active FLOP, GB, latency |
| Quantization-aware training                        | `gemma-3-technical-report`                                       | `disclosed`                                              | v0.2.0, G9                        | post-training quantization, on measured quality drift                  |
| Speculative decoding from nested slices            | `matformer`                                                      | `reported-result`                                        | v0.8.0                            | MTP heads as the competing draft source                                |
| Multi-teacher distillation                         | `gemma-3-technical-report`, `matformer`                          | `disclosed`                                              | v0.0.7, v0.2.0                    | from-scratch student at matched compute; licensed teacher outputs only |
| Constitutional critique and revision               | `anthropic-constitutional-ai`                                    | `disclosed`; benefit `hypothesis`                        | v0.0.7, G10                       | SFT and human-preference controls                                      |
| DPO baseline, then scope-limited GRPO              | `deepseek-llm`, `deepseek-math`                                  | `disclosed`                                              | v0.0.7, v0.5.0                    | SFT baseline; verifiable rewards only, never free-form as primary      |
| Cold-start data before RL, readability gates       | `deepseek-r1`                                                    | `reported-result` failure mode                           | v0.5.0                            | RL-first variant, on readability and language consistency              |
| Model-written evaluations                          | `anthropic-model-written-evals`                                  | `disclosed`; bias `reported-result`                      | v0.0.7                            | human-authored held-out set; humans keep evaluation ownership          |
| Near-duplicate control and repetition budgeting    | `anthropic-repeated-data`                                        | `reported-result`; mechanism `hypothesis`                | v0.0.3, G2                        | nominal token counts, on effective unique tokens and memorization      |
| Sparse autoencoders, circuit tracing               | `anthropic-scaling-monosemanticity`, `anthropic-circuit-tracing` | `disclosed`; completeness `unknown`                      | v0.6.0                            | nothing — evidence only, never a correctness certificate               |
| Contextual retrieval                               | `anthropic-contextual-retrieval`                                 | `reported-result`                                        | external system, any version      | never raises a model's trained or validated context length             |

Three rules govern the ledger. A `reported-result` is the other lab's
measurement under their data, scale, and hardware, and never substitutes for an
EPOR ablation. A technique that cannot be exported to GGUF is rejected at its
gate whatever its measured quality, because local execution is the product.
Adopting a mechanism requires beating its control on the stated metric, so
"DeepSeek does it" and "this is the newest approach" are not reasons and do not
appear in an adoption record.

## Data, training, and artifact contracts

- Initial proxy pretraining mixture: 55% rights-cleared general/reference
  English, 25% permissively licensed code and repository context, 15%
  math/science/technical material, and 5% high-quality multilingual text for
  robustness. Ablations may change the target mixture; every change is
  versioned.
- Preserve repository topology and dependency order for code. Scan licenses,
  credentials, generated/vendor material, and benchmark overlap before
  tokenization.
- Keep raw, normalized, filtered, rejected, and tokenized layers immutable and
  content-addressed. Derived builds reference parents rather than overwriting.
- Pure PyTorch is canonical. Transformers compatibility is an adapter/export
  contract, not dependence on `Trainer`.
- A canonical release contains sharded Safetensors, model and generation config,
  tokenizer/chat template, model card, run manifest, dataset/tokenizer/code
  hashes, license, benchmark report, and signed checksums.
- Quantized artifacts record source checkpoint, converter and llama.cpp commit,
  quantization scheme, context/cache settings, peak RAM/VRAM, TTFT, tokens/sec,
  and measured quality drift.
- Repository releases use SemVer. Recipes and checkpoints have independent,
  immutable IDs; patch releases cannot silently change architecture, tokenizer,
  data mixture, or chat format.

## Public interfaces

CLI surface:

```text
epor doctor
epor research sync|verify|index
epor config validate
epor data ingest|build|audit
epor tokenizer train|evaluate
epor train pretrain|resume|sft|preference|grpo
epor eval run|compare
epor export hf|gguf
epor serve
epor api
epor worker
epor ui
```

The control API lives under `/api/v1` and supplies health/capabilities, typed
allowlisted jobs, job listing/detail/cancellation/events/artifacts, and the
stable error envelope `{code, message, details, request_id}`.

The eventual inference API supplies `/v1/models`, `/v1/completions`, and
`/v1/chat/completions`, including streaming, cancellation, token usage,
context-limit errors, and model/profile selection. Job specs, model recipes,
dataset manifests, and evaluation suites are versioned Pydantic schemas in
YAML/JSON. UI forms consume generated JSON Schema.

## Acceptance and test gates

- **G0 — Research provenance:** All papers resolve, hashes verify, duplicate or
  version changes are visible, raw caches remain ignored, and every claim has an
  evidence classification.
- **G1 — Reproducible foundation:** A clean machine completes frozen environment
  setup, CLI doctor, database migration, UI build, and offline CPU smoke tests.
- **G2 — Data/tokenizer:** Repeated builds produce identical manifests;
  cross-source dedup works; PII/secrets and split leakage fixtures are removed;
  arbitrary bytes and code/whitespace round-trip; special-token IDs never drift.
- **G3 — Training correctness:** Reference and optimized attention agree within
  documented tolerances; a tiny model overfits a fixture; resume reproduces
  uninterrupted loss and weights; single-process and distributed loss agree.
- **G4 — γ feasibility:** Every nested slice receives training; a static ≈4B
  slice extracts with logits preserved within tolerance; PLE/conditional-memory
  variants beat compute-matched dense baselines on accelerator residency without
  unacceptable TTFT or throughput loss.
- **G5 — Scale approval:** A 1B pilot establishes loss curves, data throughput,
  MFU, GPU-hour/cost estimates, checkpoint cadence, abort thresholds, and failure
  reserve. The UI cannot provision paid compute; every full run requires explicit
  approval.
- **G6 — Context:** Validate beginning/middle/end retrieval, RULER/passkey,
  multi-document QA, long code, perplexity by position, short-context regression,
  jailbreak/prompt-injection resistance, TTFT, RAM/VRAM, and OOM behavior. Merely
  accepting 128K/256K tokens never raises `validated_max_context`.
- **G7 — MoE/runtime:** No dead experts or silent token drops; routing remains
  balanced; α matches or beats its compute-matched dense control; PyTorch/HF/GGUF
  outputs have bounded parity drift; local profiles remain below 28 GiB RAM and
  7.5 GiB VRAM.
- **G8 — UI/security:** Validate transitions, idempotent cancellation, restart
  reconciliation, concurrent-worker claiming, path containment, redaction,
  local-only binding, API errors, polling de-duplication, keyboard accessibility,
  and persisted artifact checksums.
- **G9 — Release:** Quantization quality remains within the published regression
  budget; data/license/contamination reports pass; artifacts reproduce from
  manifests; model/system cards distinguish configured, trained, validated, and
  locally certified context.
- **G10 — Safety covenant:** Test direct harm, foreseeable harm through
  inaction, unsafe or unauthorized instructions, conflicts between human
  directions, legitimate shutdown and correction, and self-preservation
  pressure. A release must fail closed or escalate to human review when the
  ordered principles cannot be satisfied with adequate confidence. The suite
  opens at v0.0.2 against the resolver and widens at each version that adds a
  new decision surface — job admission, alignment training, tool use, serving —
  so no subsystem ships before its covenant cases exist.

## Assumptions and boundaries

- Linux is the primary development/orchestration platform. Local inference uses
  GGUF/llama.cpp Vulkan with CPU fallback.
- The reference machine is a Ryzen 5-class CPU, RX 5700 XT with approximately
  8 GiB VRAM, and 32 GiB RAM.
- α/β/γ are three independently versioned public model families.
- EPOR-γ's “≈4B” means its intended core accelerator-memory path; it is not a
  claim that an 8B checkpoint becomes a 4B-active MoE.
- 128K/256K values are configured and cloud-validation targets, not promised
  interactive local windows.
- The project is English-first and text-only through v1, with code and math as
  first-class domains.
- Telemetry remains local by default through JSONL, TensorBoard, SQLite, and the
  console. External tracking is optional and disabled.
- v0.0.1 performs no autonomous training-data crawl, paid cloud provisioning,
  external-model API spending, or full-size training.
- Anthropic and DeepSeek methods are research inputs, not claims that EPOR
  reproduces Claude or DeepSeek. Proprietary corpora, trainers, and undisclosed
  architectures are recorded as `unknown`, never guessed.
