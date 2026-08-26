# DeepSeek research dossier

Source IDs refer to [`research/catalog.yaml`](../../research/catalog.yaml).
Reported efficiency and benchmark values remain results from DeepSeek's systems
until independently reproduced under EPOR manifests.

## Dense scaling, code, and math

- **`[disclosed]`** `deepseek-llm` describes dense base/chat models, a scaled
  pretraining study, supervised tuning, and Direct Preference Optimization.
- **`[EPOR-adaptation]`** EPOR uses small-to-1B scaling sweeps to select data and
  hyperparameters; a target parameter count is never approved from a paper
  result alone.
- **`[disclosed]`** `deepseek-coder` describes code-focused pretraining that
  incorporates repository-level context and fill-in-the-middle objectives.
- **`[EPOR-adaptation]`** EPOR preserves repository topology and dependency order,
  then scans licenses, secrets, generated/vendor trees, and benchmark overlap
  before tokenization. FIM tokens are reserved in the original tokenizer.
- **`[disclosed]`** `deepseek-math` describes mathematical data selection and
  introduces Group Relative Policy Optimization in its post-training research.
- **`[EPOR-adaptation]`** EPOR starts with supervised and preference baselines;
  later GRPO is limited to α and primarily uses verifiable compiler, test, or
  SymPy rewards with human-cleaned cold starts.

## Mixture of experts and latent attention

- **`[disclosed]`** `deepseek-moe` separates shared experts from fine-grained
  routed experts to encourage specialization while retaining common knowledge.
- **`[reported-result]`** The paper reports favorable quality/compute trade-offs
  for its tested routing configurations; these do not establish EPOR-α's top-k,
  capacity, balancing, or communication policy.
- **`[disclosed]`** `deepseek-v2` combines a DeepSeekMoE design with Multi-head
  Latent Attention to reduce key/value-cache cost in its architecture.
- **`[hypothesis]`** MLA may improve EPOR long-context memory efficiency, but its
  implementation complexity and PyTorch/HF/GGUF parity must beat GQA and cache
  quantization controls before adoption.
- **`[EPOR-adaptation]`** EPOR-α targets approximately 30B total and 6–8B measured
  active parameters per token. Proxy ablations must show no dead experts, no
  silent token drops, balanced routing, acceptable communication, and a win over
  a compute-matched dense control.

## Continued code training and V3

- **`[disclosed]`** `deepseek-coder-v2` extends code specialization through
  continued pretraining of a MoE family and reports expanded language and
  long-context capabilities.
- **`[EPOR-adaptation]`** EPOR treats continued pretraining as a new immutable
  lineage with regression tests against general language, safety, calibration,
  memorization, and prior code tasks.
- **`[disclosed]`** `deepseek-v3` describes an MoE system using MLA,
  auxiliary-loss-free load balancing, multi-token prediction, FP8 training, and
  substantial systems co-design.
- **`[disclosed]`** The balancing strategy adjusts a per-expert routing bias at a
  configured update speed instead of relying on an auxiliary balance loss, and
  retains only a very small sequence-wise balance term to prevent extreme
  imbalance within a single sequence. The bias update speed is described as
  decaying to zero for the final portion of training.
- **`[disclosed]`** Routing communication is bounded by sending each token to at
  most a fixed number of nodes, chosen by summed per-node expert affinity.
- **`[disclosed]`** Multi-token prediction is a training objective with its own
  loss weight, described as reduced partway through training, and keeps a full
  causal chain at each prediction depth; the paper also reports ablations for it.
- **`[disclosed]`** Training uses an FP8 mixed-precision framework with
  fine-grained quantization, paired with a pipeline schedule designed to overlap
  computation and cross-node expert communication.
- **`[reported-result]`** DeepSeek reports training efficiency and benchmark
  results for its hardware/software stack; EPOR has neither reproduced those
  values nor inferred that they transfer to a small local system.
- **`[EPOR-adaptation]`** Each of these is an independent ablation against a
  control, decided at proxy scale, after the reference loop and the distributed
  checkpoint/resume path pass correctness gates: bias-based balancing against an
  auxiliary-loss baseline, multi-token prediction against single-token training
  at matched tokens, and node-limited routing against unrestricted routing.
- **`[EPOR-adaptation]`** FP8 is not on the EPOR reference-machine path. BF16
  remains the training baseline and FP8 stays a cloud-only ablation that must
  show loss-curve agreement against a BF16 control before it is used for a run
  that produces a released checkpoint.
- **`[unknown]`** The bias update speed, MTP weight schedule, and node cap that
  suit a 671B-parameter cluster run are not known to be the right values at EPOR
  proxy or target scale; they are starting points for a sweep, not settings.

## Reasoning and V3.2

- **`[disclosed]`** `deepseek-r1` investigates reinforcement learning for
  reasoning, human-readable cold-start data, rejection sampling, and distilling
  reasoning behavior into smaller models.
- **`[reported-result]`** The paper reports gains on several reasoning tasks and
  also discusses readability/language-mixing issues in an RL-first variant;
  reported benchmark scores are not EPOR results.
- **`[EPOR-adaptation]`** EPOR builds verified math/code traces, samples both
  successes and failures for audit, limits GRPO scope, and distills only licensed
  teacher outputs into independently initialized β and γ students.
- **`[disclosed]`** `deepseek-v3-2` describes a sparse-attention mechanism and a
  scaled reasoning/agent post-training program.
- **`[disclosed]`** The mechanism pairs a lightweight scoring component, using a
  cheap activation chosen for throughput, with fine-grained selection of which
  preceding tokens each query attends to. It is described as introduced into an
  otherwise unchanged architecture through continued training rather than as a
  from-scratch design.
- **`[hypothesis]`** Sparse attention may reduce long-context cost without
  unacceptable retrieval or short-context regression; EPOR requires dense/local-
  global controls, position-wise tests, kernel portability, and export parity.
- **`[EPOR-adaptation]`** Because the mechanism is described as retrofittable by
  continued training, EPOR treats it as a post-hoc long-context option evaluated
  against an already-certified dense or local/global checkpoint, which keeps a
  failed experiment from blocking a release. A sparse-attention variant that no
  GGUF backend can execute is not a candidate regardless of measured quality.

## Conditional memory

- **`[disclosed]`** `deepseek-engram` presents conditional lookup memory as an
  additional sparsity axis intended to move some static association capacity out
  of repeatedly computed Transformer blocks.
- **`[disclosed]`** The source frames the design choice as a sparsity-allocation
  problem — how much capacity to spend on conditional computation versus static
  lookup — and reports a U-shaped relationship with an interior optimum rather
  than a monotonic preference for either extreme.
- **`[disclosed]`** Lookup addressing is described as deterministic, which is what
  permits prefetching entries from host memory instead of holding them in
  accelerator memory.
- **`[EPOR-adaptation]`** The allocation framing, not the reported optimum, is
  what EPOR adopts: γ's proxy sweep measures its own computation/lookup split
  against compute-matched dense and MatFormer controls. Deterministic addressing
  is a hard prerequisite, since a lookup that cannot be prefetched from host
  memory defeats the accelerator-residency goal that motivates γ.
- **`[reported-result]`** The source reports quality/efficiency results for its
  tested Engram configurations; they do not establish usability on EPOR's target
  architecture or storage hierarchy.
- **`[hypothesis]`** An Engram-style module could improve knowledge retained per
  active FLOP or resident accelerator byte, but lookup latency, CPU/storage
  traffic, tokenizer coupling, update semantics, and quantization may erase the
  benefit.
- **`[EPOR-adaptation]`** Conditional memory is gated behind compute-matched
  dense and MatFormer proxy baselines and must report TTFT, throughput, RAM/VRAM,
  quality, and failure behavior before it can enter EPOR-γ.

## Official infrastructure repositories

- **`[disclosed]`** `deepseek-v3-repository`, `deepseek-deepgemm`,
  `deepseek-flashmla`, and `deepseek-deepep` expose selected model/runtime code,
  matrix kernels, MLA kernels, and expert-parallel communication components.
- **`[unknown]`** Mutable public repositories do not disclose DeepSeek's complete
  internal production or training environment, deployment topology, dependency
  pins, monitoring, or private modifications.
- **`[EPOR-adaptation]`** EPOR records an exact commit, license, supported
  hardware, compiler/runtime versions, tests, and fallback before adopting code.
  Hopper-specific or provider-specific kernels remain outside the CPU-safe
  environment.

## Boundary

- **`[unknown]`** Papers cannot establish undisclosed corpus composition,
  filtering details, internal failures, or end-to-end reproducibility beyond the
  released artifacts.
- **`[EPOR-adaptation]`** DeepSeek is a research input, not a claim that EPOR
  reproduces DeepSeek models, infrastructure, data, or benchmark performance.
