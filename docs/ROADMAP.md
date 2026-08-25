# EPOR AI roadmap

This roadmap is a sequence of evidence gates, not a promise that target-scale
training has been funded. Every released EPOR weight starts from an independent
random initialization. Licensed teacher logits, synthetic examples, filtering
judgments, and verified reasoning traces may inform training; external weights
must never seed or be merged into an EPOR checkpoint.

Public names are always **EPOR-γ**, **EPOR-α**, and **EPOR-β**. Machine-safe IDs
are `epor-gamma`, `epor-alpha`, and `epor-beta`.

| Model | Purpose | Architecture destination | Configured context target | Local v1 certification goal |
|---|---|---|---:|---:|
| **EPOR-γ** | Compact, knowledge-efficient local model | About 8B total parameters; nested ≈4B accelerator-resident core; optional ≈2B slice; MatFormer-style elasticity, distillation, and gated PLE/conditional-memory research | 128K | 32K |
| **EPOR-α** | Balanced general, code, and reasoning model | 10–12B dense decoder with RMSNorm, SwiGLU, GQA, RoPE, and backend-supported local/global attention | 256K | 16K |
| **EPOR-β** | Highest-capability reasoning, code, and math model | About 30B total MoE parameters, 6–8B measured active per token, with shared and routed experts | 256K | 8K |

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
- **v0.0.2 — Provenance-first data engine:** Add rights-aware registration,
  streaming ingestion, a normalized document schema, exact and global
  near-deduplication, language/domain/quality classification, PII and credential
  removal, malware/unsafe-content quarantine, robots/terms records, removal
  tombstones, content-hash splits, and dataset cards.
- **v0.0.3 — Tokenizer and dataset v0:** Train an original 65,536-ID byte-level
  BPE with 256 reserved IDs for roles, tools, FIM, documents, retrieval, and
  future extensions. Preserve whitespace/code exactly, guarantee byte fallback,
  measure domain fertility, and write content-addressed token shards with
  document boundaries.
- **v0.0.4 — Training kernel:** Add token-based batching, AdamW (`β1=.9`,
  `β2=.95`, weight decay `.1`, clip `1.0`), warmup/cosine scheduling, sequence
  packing, mixed precision, activation checkpointing, DDP, FSDP2, distributed
  checkpoints, fault injection, and exact data-cursor/RNG resume.
- **v0.0.5 — Evaluation and scaling laboratory:** Add validation
  loss/perplexity, code/math/general benchmarks, calibration, contamination and
  memorization tests, safety suites, hardware profiling, data-mixture ablations,
  and 10M→100M→300M→1B sweeps. Project tokens, GPU-hours, storage, failure
  reserve, and cost before any target-size run.
- **v0.0.6 — Knowledge transfer and alignment:** Add teacher-data provenance,
  response/logit distillation, SFT, pairwise preferences, a DPO baseline,
  constitutional critique/revision, AI-judged preferences with human audits,
  reward-overoptimization monitoring, an original versioned EPOR constitution,
  and explicit evaluations of the safety-covenant priority order.
- **v0.0.7 — Context laboratory:** Train progressively at 4/8K→32K→128K and
  then 256K for α/β only after 128K passes. Evaluate RoPE/YaRN, local/global
  attention, GQA cache quantization, MLA and sparse-attention experiments,
  long-document mixtures, lost-middle behavior, prompt injection, TTFT,
  throughput, and memory.
- **v0.0.8 — Export, inference, and full console:** Add Hugging
  Face/Safetensors export, GGUF conversion, Q8/Q6/Q5/Q4 profiles, llama.cpp
  Vulkan integration, quantization parity reports, a provider-neutral local
  generation API, chat playground, registries for models/datasets/tokenizers/
  evaluations, a cost estimator, and artifact comparison.
- **v0.0.9 — Three-family proxy release:** Train and publish tiny/proxy
  EPOR-γ, EPOR-α, and EPOR-β checkpoints with separate model cards, scaling
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
  validation and 32K on the reference PC.
- **v0.3.0 — EPOR-α:** Train the 10–12B dense model with the selected data
  mixture and scaling-law hyperparameters. Target a practical Q4 Vulkan
  artifact, 256K cloud validation, and a 16K local profile.
- **v0.4.0 — EPOR-β:** Train the ≈30B-total MoE with shared and fine-grained
  routed experts. Proxy ablations select top-k routing and balancing policy.
  Require 6–8B measured active parameters, no dead experts, acceptable routing
  communication, GGUF parity, 256K cloud validation, and an 8K local profile.
- **v0.5.0 — Reasoning specialization:** Build human-cleaned cold starts,
  verified math/code traces, rejection sampling, compiler/test/SymPy rewards,
  and limited GRPO for β. Distill verified β reasoning into α and γ. Do not use
  unverifiable free-form rewards as the primary RL signal.
- **v0.6.0 — Safety and interpretability:** Expand the EPOR constitution and its
  ordered safety covenant, held-out red teaming, input/output classifiers,
  calibration and abstention, activation hooks, probes, sparse autoencoders,
  and causal interventions. Treat interpretability as evidence, not a
  correctness certificate.
- **v0.7.0 — Context certification:** Complete position-wise and task-level
  validation for γ at 128K and α/β at 256K on cloud hardware. Certify lower
  operational profiles on the exact Ryzen/RX 5700 XT/32 GiB reference system.
- **v0.8.0 — Advanced efficiency:** Evaluate MLA, DeepSeek Sparse Attention,
  multi-token prediction/speculative decoding, Engram-style conditional memory,
  quantization-aware training, expert streaming, and Colibri integration.
  Colibri remains optional until measured usable. **`[disclosed]`** Its official project
  demonstrates disk-streamed MoE execution with architecture-specific runtime
  work ([Colibri](https://github.com/JustVugg/colibri)).
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
  balanced; β matches or beats its compute-matched dense control; PyTorch/HF/GGUF
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
  ordered principles cannot be satisfied with adequate confidence.

## Assumptions and boundaries

- Linux is the primary development/orchestration platform. Local inference uses
  GGUF/llama.cpp Vulkan with CPU fallback.
- The reference machine is a Ryzen 5-class CPU, RX 5700 XT with approximately
  8 GiB VRAM, and 32 GiB RAM.
- γ/α/β are three independently versioned public model families.
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
