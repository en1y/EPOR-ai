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
  later GRPO is limited to β and primarily uses verifiable compiler, test, or
  SymPy rewards with human-cleaned cold starts.

## Mixture of experts and latent attention

- **`[disclosed]`** `deepseek-moe` separates shared experts from fine-grained
  routed experts to encourage specialization while retaining common knowledge.
- **`[reported-result]`** The paper reports favorable quality/compute trade-offs
  for its tested routing configurations; these do not establish EPOR-β's top-k,
  capacity, balancing, or communication policy.
- **`[disclosed]`** `deepseek-v2` combines a DeepSeekMoE design with Multi-head
  Latent Attention to reduce key/value-cache cost in its architecture.
- **`[hypothesis]`** MLA may improve EPOR long-context memory efficiency, but its
  implementation complexity and PyTorch/HF/GGUF parity must beat GQA and cache
  quantization controls before adoption.
- **`[EPOR-adaptation]`** EPOR-β targets approximately 30B total and 6–8B measured
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
- **`[reported-result]`** DeepSeek reports training efficiency and benchmark
  results for its hardware/software stack; EPOR has neither reproduced those
  values nor inferred that they transfer to a small local system.
- **`[EPOR-adaptation]`** Multi-token prediction, FP8, custom kernels, and
  auxiliary-loss-free balancing remain independent optional ablations after the
  reference loop and distributed checkpoint/resume path pass correctness gates.

## Reasoning and V3.2

- **`[disclosed]`** `deepseek-r1` investigates reinforcement learning for
  reasoning, human-readable cold-start data, rejection sampling, and distilling
  reasoning behavior into smaller models.
- **`[reported-result]`** The paper reports gains on several reasoning tasks and
  also discusses readability/language-mixing issues in an RL-first variant;
  reported benchmark scores are not EPOR results.
- **`[EPOR-adaptation]`** EPOR builds verified math/code traces, samples both
  successes and failures for audit, limits GRPO scope, and distills only licensed
  teacher outputs into independently initialized α and γ students.
- **`[disclosed]`** `deepseek-v3-2` describes a sparse-attention mechanism and a
  scaled reasoning/agent post-training program.
- **`[hypothesis]`** Sparse attention may reduce long-context cost without
  unacceptable retrieval or short-context regression; EPOR requires dense/local-
  global controls, position-wise tests, kernel portability, and export parity.

## Conditional memory

- **`[disclosed]`** `deepseek-engram` presents conditional lookup memory as an
  additional sparsity axis intended to move some static association capacity out
  of repeatedly computed Transformer blocks.
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
