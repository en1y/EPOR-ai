# Training kernel

Version 0.0.5 implements EPOR's pretraining kernel: packed token-shard
batching, AdamW with warmup and cosine decay, BF16 autocast, gradient
accumulation, activation checkpointing, DDP and FSDP2, distributed checkpoints,
fault injection, and exact data-cursor and RNG resume. It reads the token
shards written by the [tokenizer](TOKENIZER.md) and runs entirely on the local
CPU: nothing is downloaded, published, served, or sent to a model API, and no
paid compute is reachable from it.

**Scope of the evidence.** Every kernel run so far is a local proxy-scale run of
a tiny recipe on a fixture or proxy-scale build. Losses and perplexities are
measurements of a named build split, comparable only between runs on the same
build, and none is a trained or validated model result. QK-norm and multi-token
prediction (MTP) are implemented behind flags; whether either pays for itself is
undecided until G3/G5 compares each against its control at matched tokens, and
no benefit is claimed here. FP8 is not implemented.

The v0.0.1 debug path (`epor train pretrain|resume`, `epor eval run`) on the
byte fixture is separate and unchanged; CI still runs it as the offline smoke.

## Commands

```bash
# Train on a tokenized build (fp32 here; bf16 is the default).
uv run --frozen --no-sync epor train kernel configs/models/epor-tiny.yaml \
  --tokenized-id <tokenized-id> --output runs/kernel-tiny --max-steps 200 \
  --vocab-from-build --precision fp32 --seq-len 128 --tokens-per-batch 4096 \
  --micro-batch-size 8 --learning-rate 3e-3 --warmup-steps 20 --checkpoint-every 50

# The same run across two local loopback-only ranks.
uv run --frozen --no-sync epor train kernel ... --strategy fsdp2 --world-size 2

# Continue a run exactly (any strategy or world size), then evaluate it.
uv run --frozen --no-sync epor train kernel-resume runs/kernel-tiny --max-steps 400
uv run --frozen --no-sync epor eval kernel runs/kernel-tiny --split test
```

Every command resolves its covenant declaration (`kernel_train` or
`kernel_evaluate`) before touching anything, keeps recipe, output, and
checkpoint paths beneath `EPOR_PROJECT_ROOT`, reads shards from the private data
root, and prints JSON. Exit codes: `0` success, `2` invalid input or a failed
verification (missing or tampered build or checkpoint, a resume that changes
the trajectory, an out-of-bounds setting), `3` covenant refusal, `130` a
checkpointed stop, `1` any other failure.

The first SIGINT or SIGTERM asks every rank to finish its current optimizer
step, checkpoint it, and stop (exit `130`); a second signal aborts immediately.
`kernel-resume` then continues from the last checkpoint.

The control plane offers the same work as `kernel_train` and `kernel_evaluate`
jobs, and the console's Training page lists every verified run (see
[Operator surfaces](#operator-surfaces)).

### Model recipes and vocabulary

A run's model vocabulary must equal the tokenized build's. Shipped recipes are
never edited to fit a build: the 260-ID `epor-tiny` recipe exists for the debug
byte tokenizer, while every v0.0.4 tokenizer has at least 520 IDs.
`--vocab-from-build` (`vocab_from_build` in a job) trains a **derived** config
whose `vocab_size` and declared parameter counts come from the build. The
derived config has its own canonical hash, which the run manifest and every
checkpoint record; the recipe file and its hash are unchanged. Without the flag
a mismatch is refused (exit `2`) with a message naming the flag.

`--qk-norm/--no-qk-norm` and `--mtp-depth` derive a config the same way, so an
ablation trains the recipe's control block and the variant from the same file.
The kernel accepts dense decoders of at most 50M parameters, at most 100,000
optimizer steps, and at most 1,048,576 tokens per step
(`epor.reference_policy.validate_kernel_model|validate_kernel_run`).

## Settings

`PackedTrainingSettings` splits settings into two kinds.

**Trajectory** settings decide the numbers a step computes and are fixed for a
lineage; a resume that changes any of them is refused, naming the fields:
`tokenized_id`, `train_split`, `seq_len`, `tokens_per_batch`, `data_seed`,
`model_seed`, `precision`, and the optimizer (`learning_rate`, `min_lr_ratio`,
`warmup_steps`, `total_steps`, `beta1`, `beta2`, `eps`, `weight_decay`,
`gradient_clip`). The model config is checked by hash in the same way.

**Runtime** settings change speed, memory, or bookkeeping only and may differ
on resume: `micro_batch_size`, `activation_checkpointing`, `checkpoint_every`,
`keep_checkpoints`, `validation_split`, and `eval_max_steps`. World size and
strategy may also change.

| Setting | Default | Notes |
| --- | --- | --- |
| Optimizer | AdamW, `β1=0.9`, `β2=0.95`, weight decay `0.1`, `eps=1e-8` | Roadmap values; `eps` is PyTorch's default **`[EPOR-adaptation]`**. |
| Decay groups | every parameter with ≥2 dimensions | Linear weights and the (tied) embedding; norm gains never decay. Tied parameters appear once. |
| Clipping | global L2 norm `1.0` | Accumulated in float64; see [FSDP2](#ddp-and-fsdp2-on-cpu). |
| Schedule | linear warmup, cosine to `min_lr_ratio · peak` at `total_steps` | A pure function of the step, so resume needs no scheduler state. The CLI defaults `total_steps` to `--max-steps`; set it explicitly to plan a run that will be resumed past its first stop. |
| Precision | `bf16` | CPU autocast around the forward; FP32 parameters, gradients, optimizer state, and loss. `fp32` disables autocast. |

## One optimizer step

1. Reseed the global torch RNG from `(model_seed, step, rank)`, so anything
   random inside a step is a pure function of the step and survives resume.
2. Draw this rank's micro-batches from the packed stream.
3. Backpropagate each micro-batch's share of the global-step objective.
   Non-final micro-batches skip cross-rank gradient reduction (`no_sync` for
   DDP, `set_requires_gradient_sync(False)` for FSDP2).
4. Clip the global gradient norm, set the scheduled learning rate, step AdamW.
5. Gather per-micro-batch loss sums in global micro-batch order and sum them in
   that order, so every topology that computes the same per-micro-batch values
   logs the same number.

### Loss normalization

Loss is normalized by the real target tokens of the **global** step, never per
micro-batch. A micro-batch contributes
`main_loss · target_tokens / global_target_tokens`, plus with MTP
`mtp_loss_weight / D · Σ_d mtp_loss_d · valid_d / global_valid_d`, where
`global_valid_d` (valid MTP targets at depth `d`, masked exactly as the decoder
masks them) is gathered from every rank before the first backward. Each term is
multiplied by `world_size` because DDP and FSDP2 average gradients. The summed
gradient therefore equals the gradient of the single-batch loss over the whole
global step, independent of world size and accumulation
(`test_mtp_objective_is_exact_under_accumulation`).

The logged `training_loss` is `main_loss + mtp_loss_weight · mean_d(mtp_loss_d)`;
`main_loss` and the per-depth `mtp_losses` are logged separately, and
perplexity is always computed from the main loss alone.

### Packing

Documents are framed `[BOS, content, EOS]` and packed greedily into rows of
`seq_len` positions in a per-epoch order drawn from `(data_seed, epoch)`.
Segment IDs keep attention document-causal and RoPE positions restart per
segment. `tokens_per_batch` is the global step budget in positions and must be a
multiple of `seq_len`; every rank plans the whole step and takes its contiguous
block of rows, so the rank-ordered union of all micro-batches is exactly the
single-process batch. The data state is the cursor `(epoch, position, offset)`
plus a hash pinning the epoch order. See the `epor.training.packing` module
docstring for the full contract.

## DDP and FSDP2 on CPU

Both strategies use the `gloo` backend on CPU and keep FP32 parameters,
gradients, and reductions; BF16 is applied by autocast identically for every
strategy. FSDP2 (`fully_shard`) wraps every decoder block and every MTP depth's
block, then the root, which keeps the embedding, final norm, MTP projections,
and the tied output head together so the tie survives sharding. Verified on
PyTorch 2.13 (CPU build).

`epor train kernel --strategy ddp|fsdp2 --world-size N` (N ≤ 8) spawns N local
ranks with `launch_local` and divides the machine's cores between them. The
launcher opens **no listener beyond loopback**: ranks rendezvous through a
`FileStore` in a private `0700` temporary directory rather than a `TCPStore`
(whose server binds every interface whatever address it is given), and gloo's
pair sockets are pinned to the loopback interface with `GLOO_SOCKET_IFNAME`.
`test_local_launch_listens_on_loopback_only` reads `/proc` while a two-rank
group is live and fails on any other listener. `init_distributed` applies the
same interface pin for an external `env://` launcher whose `MASTER_ADDR` is
loopback, but that launcher's own `TCPStore` is outside EPOR's control; no EPOR
surface starts one.

Findings while making the topologies agree:

- **Float64 clipping.** PyTorch's `clip_grad_norm_` reduces FP32 shard norms
  in a topology-dependent order, which made FSDP2 and single-process
  trajectories differ in the last FP32 bit whenever clipping was active. The
  kernel accumulates per-parameter sums of squares in float64 in parameter
  order (all-reducing sharded partial sums in one collective), so every
  topology computes the same clip coefficient **`[EPOR-adaptation]`**.
- **Exact versus within tolerance.** When each rank's micro-batch is one of the
  single process's micro-batches, the arithmetic is identical and FP32 DDP and
  FSDP2 agree with one process **bitwise**, losses and final weights. With two
  accumulation micro-batches per rank the gradient sum associates differently
  (`(g0+g1)+(g2+g3)` versus `((g0+g1)+g2)+g3`), so BF16 + QK-norm + MTP +
  activation checkpointing agrees within `rel=1e-5, abs=1e-5` on losses and
  `1e-5` on weights.
- **Dropout.** The per-step seed includes the rank, so dropout masks depend on
  the topology. Cross-topology agreement assumes `dropout = 0`, which every
  shipped recipe uses; exact resume within one topology holds either way.

## Checkpoints and exact resume

```text
<output>/
  manifest.json                 # live run manifest (rank 0)
  manifests/<run_id>.json       # immutable copy written when the run ends
  metrics.jsonl                 # one record per optimizer step
  evaluations/                  # `epor eval kernel` reports
  checkpoints/
    latest.json                 # {"name", "step", "checkpoint_sha256"}
    step-NNNNNNNN/
      state/                    # torch.distributed.checkpoint: model + optimizer
      rng-rankNNNNN.pt          # each rank's torch CPU and Python RNG state
      trainer.json              # step, counters, data cursor, settings, config
      checkpoint.json           # sha256 and size of every file above, written last
```

Model and optimizer state go through `get_state_dict`/`set_state_dict`, so keys
are plain module names whatever the wrapper, and DCP reshards on load: a
checkpoint saved by one strategy or world size loads into another, including a
single process. Every rank writes into a temporary directory; rank 0 writes the
trainer state and the hashed manifest, renames the directory into place, and
only then replaces `latest.json`. A crash at any earlier point leaves
`latest.json` naming the previous complete checkpoint, and abandoned temporary
directories are removed at the next start. Loading re-verifies every file hash
before any tensor is read. Retention keeps the newest `keep_checkpoints`.

**The exact-resume contract.** Resuming from a checkpoint reproduces the
uninterrupted run's per-step losses and final weights bitwise, provided that:
the model config hash and every trajectory setting match (enforced), the
tokenized build verifies against its hashes (enforced), the data pipeline
version and epoch-order hash match (enforced), and the run uses the same
PyTorch build and thread-independent CPU kernels (deterministic algorithms are
switched on for the run). Across topologies the agreement is exact or within
the tolerance described above. Resuming into the same output directory requires
the same lineage; a new output directory starts a child lineage record that
names its parent run and checkpoint.

## Fault injection

`FaultPlan(step, point, mode, ranks)` fails a run at a chosen optimizer step,
either after the update and before that step's checkpoint (`point="step"`) or
mid-checkpoint after every rank's state is on disk and before promotion
(`point="checkpoint_write"`); `mode="raise"` raises `InjectedFault`, and
`mode="exit"` kills the process with no cleanup. It is a test instrument only:
it is passed explicitly to `pretrain_packed`, `fault_plan_from_env` reads the
environment only when called, and **no operator surface accepts one** — not the
CLI, not the job API. The covenant declaration for `kernel_train` states this.
Tests use it to show that a crash between checkpoints and a killed distributed
checkpoint write both recover to the uninterrupted result
(`test_crash_recovery_reproduces_uninterrupted_run`,
`test_killed_distributed_checkpoint_write_recovers_on_one_process`).

## Evaluation

`evaluate_packed_checkpoint` (`epor eval kernel`, `kernel_evaluate` jobs) runs
one unshuffled pass over a split in a single process with the checkpoint's own
model and geometry and reports token-weighted main loss, perplexity from main
loss only, per-depth MTP losses, target tokens, and batches. Evaluating the
train split is flagged as in-sample. A training run also evaluates its
validation split at the end unless `--validation-split none`.

## Operator surfaces

- **CLI:** `epor train kernel`, `epor train kernel-resume`, `epor eval kernel`
  (above). Local DDP and FSDP2 are CLI-only.
- **Control jobs:** `kernel_train` and `kernel_evaluate` take bounded typed
  specs. Jobs train in **one process**: the spec accepts only
  `strategy: none`, because spawning a process group from inside the worker
  thread would need its own cancellation and progress channels and is not
  needed for proxy-scale runs. Jobs report progress every optimizer step,
  cancel cooperatively after checkpointing the last completed step, and may
  resume from another job's run only when the trainer confirms the trajectory
  is unchanged. Paths are contained beneath the artifact root.
- **Read API:** `GET /api/v1/training/runs` lists kernel runs from job outputs
  and the CLI `runs/` directory, newest first, with the latest checkpoint
  verified against its pointer digest, recorded file sizes, and trainer-state
  hash, and with finished manifests checked against their immutable copy. Full
  tensor hashes are re-verified whenever a checkpoint is loaded. A run or report
  that fails is listed as an integrity error, not shown as healthy.
- **Console:** the Training page shows each run's step, losses, MTP losses,
  tokens seen and throughput, precision, topology, features, checkpoint, and
  validation result, labelled as proxy-scale measurements, and prefills a
  training job from the most recent tokenized build.

Job results keep `tokens_seen`, `tokens_per_batch`, and `target_tokens`
legible: they are exact keys on the redaction allowlist for language-model
token measurements, and their values are still scanned for inline credentials.

## Limitations

- Dense decoders only, at most 50M parameters, on the CPU reference machine.
  Nothing here has run on a GPU or at target scale.
- Cross-topology agreement assumes `dropout = 0` (see above).
- Per-step loss records and MTP target counts travel by `all_gather_object`,
  which pickles Python objects through the process group. That is cheap at the
  local world sizes EPOR runs and would be a bottleneck at large scale.
- A document longer than the space left in a row continues as a new segment:
  RoPE positions restart at 0 and the chunk cannot attend to the document's
  earlier tokens. Long documents are trained as independent windows.
- Throughput is wall-clock over optimizer steps on this machine, excluding
  checkpoint and evaluation time; it is not MFU and does not predict GPU speed.
- FP8 mixed precision is not implemented. It remains a cloud-only ablation in
  the roadmap, gated on loss-curve agreement with a BF16 control, and never the
  reference-machine path.
- A process killed without a stop request leaves its live manifest at
  `running`; its last complete checkpoint is still valid for resume.
