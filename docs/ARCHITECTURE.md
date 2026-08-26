# Architecture

EPOR AI separates immutable evidence and artifacts from rebuildable indexes and
control state. v0.0.1 is deliberately a tiny, CPU-safe proof of the contracts
needed for later training; none of the target-family parameter counts represent
weights present in this repository.

## Model families

All model weights begin from independent random initialization. Teacher models
may provide licensed signals or synthetic data, but their parameters are never
copied, merged, or used as checkpoint initialization.

Families are listed alphabetically, and intended capability descends in the same
order: α is the flagship, γ the compact local model.

| Family                | Planned structure                                                                         | Parameter accounting                                  |                                Context target |
|-----------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------|----------------------------------------------:|
| EPOR-α (`epor-alpha`) | Shared plus fine-grained routed experts                                                   | ≈30B total; 6–8B active measured per token            |  256K configured; 8K local certification goal |
| EPOR-β (`epor-beta`)  | Dense RMSNorm/SwiGLU/GQA/RoPE decoder with backend-supported local/global attention       | 10–12B total and active                               | 256K configured; 16K local certification goal |
| EPOR-γ (`epor-gamma`) | Elastic dense decoder with nested slices; optional PLE and conditional-memory experiments | ≈8B total, ≈4B core resident path, optional ≈2B slice | 128K configured; 32K local certification goal |

Every recipe keeps these concepts separate:

- `configured_max_context`: the architecture/configuration ceiling.
- `trained_max_context`: the longest sequence length represented in training.
- `validated_max_context`: the longest length that passed the complete context
  evaluation gate.
- `operational_default_context`: a conservative default for a given runtime.
- certified profiles: results measured on an exact hardware/software tuple.

Accepting a tensor of a particular length updates none of the validation fields.

## Reference model

The canonical implementation is pure PyTorch. The v0.0.1 decoder provides
RMSNorm, SwiGLU, grouped-query attention, RoPE, causal masking, tied token
embeddings, deterministic initialization, and FP32 CPU execution. A generated
corpus and debug tokenizer make smoke tests network-free. Transformers is an
interoperability/export boundary, not the canonical training loop.

Target configurations may be constructed on PyTorch's `meta` device for
parameter accounting. Full parameter allocation or training is prohibited
without the compute gates in [COMPUTE_POLICY.md](COMPUTE_POLICY.md).

## Artifact flow

```text
tracked source ledger ──sync──> ignored immutable raw files
        │                              │
        │                              └──extract──> ignored text + index
        │
data registrations ──> raw ──> normalized ──> filtered ──> token shards
                                                        │
model recipe + run spec + shards ──> checkpoint + manifest + metrics
                                                        │
                                  HF/Safetensors ──> GGUF + parity report
```

The ledger, recipes, and manifests are authoritative. SQLite is a rebuildable
query index. Raw research, datasets, run outputs, and weights stay ignored until
an explicit release process creates independently licensed artifacts.

## Research archive safety

The research synchronizer accepts only URLs already present in the validated
catalog. It enforces HTTPS, exact allowlisted hosts, bounded redirects,
content-type and byte limits, transfer-byte hashing, file signatures, atomic
replacement, path containment, and optional offline operation. Immutable
artifacts use raw-byte pins. Explicitly reviewed dynamic HTML sources instead
pin normalized document or semantic-article text under a versioned profile while
recording every raw transfer digest in ignored metadata. The control API keeps
observed raw, tracked raw, and tracked content digests distinct. A digest
observed only during local synchronization is `unpinned`; only a raw or
normalized-content digest in the reviewed tracked ledger can produce `verified`
status.

## Training and scale-out

The single-process loop remains the correctness oracle. Future distributed work
uses composable FSDP2 and distributed checkpointing, with DDP and exact RNG/data
cursor resume tested before scale. Provider-specific CUDA, ROCm, JAX, Triton,
and Hopper stacks live in separate locked environments.

The executable v0.0.1 reference path shares one allocation-free policy across
the CLI and control worker. It rejects non-reference or over-50M recipes before
model construction, caps the in-memory corpus and trusted checkpoint before
loading, and bounds steps, evaluation batches, and batch tokens. Direct CLI
paths additionally remain under `EPOR_PROJECT_ROOT`; target-family recipes are
metadata/meta-device planning inputs and cannot enter this runtime.

The reference RX 5700 XT is an inference target through llama.cpp/Vulkan with a
CPU fallback. It is not a required PyTorch training device.

## Local control plane

The API, worker, and UI are separate processes. Only typed job specifications
for `system_probe`, `research_sync`, `tiny_train`, and `tiny_eval` are accepted
in v0.0.1. The API never evaluates a client command, fetches an arbitrary URL,
or exposes a general filesystem browser.

Admission is the covenant chokepoint. `JobService.create_job` resolves the job
type against `configs/covenant-v1.yaml` through `epor.actions` before parsing
its specification, records the verdict and covenant hash in the append-only
event stream, and raises rather than queueing anything the covenant does not
permit. The worker re-resolves at dispatch so a job admitted under an earlier
covenant cannot execute under a later one. The CLI shares that registry rather
than keeping its own, so `research sync`, `train pretrain`, `train resume`,
`eval run`, and `generate` are governed identically despite never touching the
control plane. They also share the reference-runtime limits described above,
so the declaration's resource assumptions hold on either route. Stopping work
is never gated: the second principle outranks the
third, so cancellation, correction, and shutdown bypass the gate by design.

Jobs use compare-and-set transitions and append-only events. File manifests are
the durable artifact truth; the loopback SQLite database uses WAL mode for local
coordination and can be rebuilt. The server binds to `127.0.0.1`, uses explicit
origins, redacts secrets, and resolves all artifact paths under configured roots.

## Release boundaries

A release checkpoint consists of sharded Safetensors, configs, tokenizer and
chat format, model card, manifest lineage, data/tokenizer/code hashes, license,
benchmark report, and signed checksums. Quantized derivatives additionally pin
the converter and llama.cpp revisions and report quality drift plus measured
RAM, VRAM, TTFT, and throughput.

See [ROADMAP.md](ROADMAP.md) for gates and
[DATA_GOVERNANCE.md](DATA_GOVERNANCE.md) for rights and lineage policy.
