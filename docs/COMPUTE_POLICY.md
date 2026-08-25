# Compute policy

EPOR uses evidence-gated compute. A configuration file or UI card is not
authorization to allocate hardware, spend money, or train a target-size model.

## Safe default

v0.0.1 runs CPU-safe tiny models, metadata-only target parameter counts, offline
fixtures, and local loopback services. It performs no paid provisioning, no
external-model API spending, and no 8B/12B/30B training. The UI cannot provision
compute.

## Approval ladder

1. **Tiny correctness (10–50M):** overfit a fixture, pass deterministic resume,
   validate manifests, and measure reference attention.
2. **Scaling laboratory (10M→100M→300M):** establish data throughput, loss
   curves, failure modes, and architecture ablations.
3. **One-billion pilot:** measure MFU, networking and storage throughput,
   checkpoint/recovery time, context memory, and real GPU-hours.
4. **Target proposal:** project tokens, GPU-hours, provider SKU/count, storage,
   egress, evaluation, checkpoint cadence, failure reserve, carbon/energy data
   where available, and a hard budget/abort envelope.
5. **Explicit approval:** a named human approves the exact immutable proposal.
   Material change in provider, scale, duration, architecture, data mixture, or
   budget requires a new approval.

## Mandatory abort conditions

Stop or decline a run when loss is non-finite or outside the approved envelope,
data rights change, contamination or secret leakage is found, checkpoint restore
fails, expert routing drops tokens or produces dead experts, throughput makes the
budget invalid, safety monitoring detects an unapproved risk, or spend/energy/
time reaches the approved limit.

Workers may cancel approved jobs and preserve diagnostic artifacts. They may not
expand scope, increase budget, switch to paid infrastructure, or weaken an abort
threshold autonomously.

## Hardware profiles

Provider environments pin driver, runtime, framework, collective library,
kernel packages, image digest, and hardware SKU together. JAX and
accelerator-specific kernels stay separate from the default environment.

The reference Ryzen 5/RX 5700 XT/32 GiB machine certifies only explicitly
measured local profiles. The RX 5700 XT is expected to serve GGUF through
llama.cpp/Vulkan with CPU fallback; it is not a required PyTorch training GPU.

## Reports

Every run manifest records parent run, code/config/data/tokenizer hashes, seeds,
hardware, dependency lock, precision, timestamps, checkpoints, metrics, actual
resource use, and termination reason. A scaled or paid run additionally records
its required approval ID; tiny local runs may leave that field absent or null.
Failed and aborted runs are retained in scaling analysis because excluding them
would understate cost and risk.
