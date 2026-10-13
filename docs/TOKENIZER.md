# Original tokenizer and token shards

Version 0.0.4 implements EPOR's original byte-level BPE tokenizer, the
content-addressed token shards that the training kernel will read, and a
fertility report that measures both on a held-out split. Everything here runs
offline over builds produced by the [data engine](DATA_ENGINE.md); nothing is
downloaded, published, or sent to a model API.

**Scope of the evidence.** Every tokenizer, shard, and report produced so far
comes from local fixture or proxy-scale builds. No target-scale tokenizer has
been trained. The 65,536-ID layout is a fixed contract
**`[EPOR-adaptation]`**; whether 65,536 IDs are the right size for the target
mixture is a **`[hypothesis]`** that only a target-scale corpus and the v0.0.6
scaling laboratory can test. Fertility numbers are measurements of the named
build split, not validated results, and none of them predict full-scale
fertility.

## Vocabulary layout

The layout is frozen and versioned as `epor-reserved-v1` in
`src/epor/tokenizer/spec.py`. Every artifact pins the SHA-256 of the reserved
table, so a changed entry cannot load against an older artifact.

| IDs | Contents |
| --- | --- |
| `0..255` | Reserved control tokens (table below). Never produced from input text. |
| `256..511` | One token per byte, in byte order: byte `b` is ID `256 + b`. |
| `512..` | Learned merges in merge order, up to the declared vocabulary size. |

Byte fallback is therefore structural: every byte sequence, including invalid
UTF-8, encodes and decodes back to the exact source bytes. There is no
normalizer and no unknown token; whitespace and code are preserved exactly.
The pre-tokenizer is the byte-level GPT-2 split pattern with no prefix space,
so merges never cross word, punctuation, or whitespace-run boundaries and can
never spell a `<|...|>` control string.

| ID | Token | Category | Purpose |
| --- | --- | --- | --- |
| 0 | `<\|pad\|>` | control | Padding; matches the model configs' `pad_token_id`. |
| 1 | `<\|bos\|>` | control | Beginning of sequence. |
| 2 | `<\|eos\|>` | control | End of sequence. |
| 3 | `<\|system\|>` | chat | Opens a system-role message. |
| 4 | `<\|user\|>` | chat | Opens a user-role message. |
| 5 | `<\|assistant\|>` | chat | Opens an assistant-role message. |
| 6 | `<\|tool\|>` | chat | Opens a tool-role message. |
| 7 | `<\|end_turn\|>` | chat | Closes any role message. |
| 8 | `<\|tool_call_start\|>` | tool | Opens a structured tool call. |
| 9 | `<\|tool_call_end\|>` | tool | Closes a structured tool call. |
| 10 | `<\|tool_result_start\|>` | tool | Opens a tool result. |
| 11 | `<\|tool_result_end\|>` | tool | Closes a tool result. |
| 12 | `<\|fim_prefix\|>` | fim | Fill-in-the-middle prefix marker. |
| 13 | `<\|fim_middle\|>` | fim | Fill-in-the-middle middle (target) marker. |
| 14 | `<\|fim_suffix\|>` | fim | Fill-in-the-middle suffix marker. |
| 15 | `<\|doc_start\|>` | document | Document boundary: start. |
| 16 | `<\|doc_end\|>` | document | Document boundary: end. |
| 17 | `<\|retrieval_start\|>` | retrieval | Opens retrieved context. |
| 18 | `<\|retrieval_end\|>` | retrieval | Closes retrieved context. |
| 19–255 | `<\|reserved_N\|>` | reserved | Future extensions; `N` is the token's own ID. |

Text that literally contains a control string such as `<|bos|>` encodes as
ordinary bytes and merges. A reserved ID enters a sequence only when a caller
inserts it by ID; if encoding ever produced one from content, the codec fails
closed rather than returning it.

## Training

`epor tokenizer train <dataset-id> <build-id>` trains from **only the train
split** of one data-engine build. Validation and test text never reaches the
trainer. Because a build already applies rights eligibility, sensitive-value
removal, quarantine, tombstones, and deduplication, the tokenizer learns only
from training-eligible reviewed text. If any train-split document has been
tombstoned since the build was made, training refuses; rebuild the dataset
first so the artifact's pinned split hash names exactly the text it learned
from.

Training is deterministic. It runs single-threaded, and the trained result is
rewritten into the fixed layout above and serialized as canonical JSON, so the
same build and settings produce a byte-identical artifact and the same ID.
`--vocab-size` is the **declared** size (default 65,536). When a small corpus
supports fewer merges, the manifest records the smaller **learned** size next to
the declared one; nothing is padded and the declared size is never reported as
learned.

## Storage layout

All three layers live beneath the private data root (`.epor/data` by default,
or `EPOR_DATA_ROOT`) and are ignored by Git.

- `tokenizers/<tokenizer_id>/` holds `tokenizer.json` and `manifest.json`. The
  ID is the SHA-256 of the canonical manifest, which pins the `tokenizer.json`
  hash, the reserved-spec hash, the trainer settings and library version, and
  the source build, train-split hash, document count, and byte count. Loading
  re-verifies every hash and the full layout.
- `tokenized/<tokenized_id>/` holds `manifest.json`, `shards/<sha256>.u16`, and
  `indexes/<sha256>.json`. Shards are flat little-endian `uint16` token arrays.
  Each document is written as `<|doc_start|>` + content + `<|doc_end|>`; those
  are the only reserved IDs in a shard, and BOS/EOS framing is left to the
  training loader. Documents are packed in stable split order and never split
  across shards unless one document alone exceeds `--max-shard-tokens`, in which
  case its index entries record an explicit `part`/`part_count` continuation.
  The manifest pins the tokenizer, build, per-split hashes, shard and index
  hashes, the shard limit, and the tombstones applied. A build is staged under
  `tmp/` and renamed into place only when complete, so cancellation or failure
  leaves no partial artifact, and a repeated build is a verified no-op.
- `evaluations/<report_id>.json` holds one canonical fertility report, addressed
  by its own SHA-256.

## Fertility and round-trip

`epor tokenizer evaluate <tokenizer-id> <dataset-id> <build-id>` measures the
`validation` split by default (`--split test` or `--split train` on request).
The report records documents, bytes, characters, whitespace-delimited words,
content tokens, tokens per byte, tokens per word, bytes per token, and the
byte-fallback rate, overall and per domain and language label. It also counts
documents whose tokens do not decode to their exact source bytes.

- A non-zero `round_trip_failures` fails the gate: the CLI exits `1` and the
  control job fails, while the report is kept for review.
- `in_sample` is true when the split is the tokenizer's own training split. The
  CLI warns on stderr, the job logs a warning, and the console marks the report.
- Every report carries a fixed `scope` statement that it describes the named
  split only and is not a target-scale result.

## Removal

Shards and fertility reports re-apply every current tombstone with the data
engine's own matching rule, including tombstones recorded after the parent
build, and record the excluded count and tombstone IDs. A removal does not
rewrite existing artifacts, which are immutable. To act on one:

1. record the tombstone with `epor data remove`;
2. call `epor.tokenizer.affected_tokenized_builds(data_root, document_id)` to
   list every stored tokenized build, split, and shard containing the document;
3. rebuild the dataset, retrain the tokenizer if the document was in its train
   split, and re-run `epor tokenizer shard`, which produces a new tokenized ID
   without the document;
4. retire the superseded tokenized builds from any training manifest.

Retiring and deleting superseded artifacts is an operator step; v0.0.4 does not
delete data automatically.

## Commands, jobs, and console

```bash
uv run --frozen --no-sync epor tokenizer train my-dataset-v1 <build-id>
uv run --frozen --no-sync epor tokenizer shard my-dataset-v1 <build-id> <tokenizer-id>
uv run --frozen --no-sync epor tokenizer evaluate <tokenizer-id> my-dataset-v1 <build-id>
```

Each command resolves its reviewed covenant declaration (`tokenizer_train`,
`tokenizer_shard`, `tokenizer_evaluate`) before reading the data root, prints
JSON, and exits `2` with a refusal message when an input is missing or fails
verification. Repeating a command with the same inputs returns the same ID.
The tokenizer modules need the optional `data` dependency profile
(`uv sync --frozen --extra data`).

The same operations are typed control jobs with cooperative cancellation and a
JSON report artifact. `GET /api/v1/tokenizers` lists tokenizers, tokenized
builds, and fertility reports after checking each one's content address;
anything that fails is listed under `integrity_errors` instead. That listing
checks shard manifests, not every shard byte; `verify_tokenized_build` performs
the full shard check. The console's **Tokenizer** page shows declared and
learned vocabulary separately, shard counts and exclusions, and per-domain and
per-language fertility with the report's scope, round-trip failures, and any
in-sample warning.

## Boundaries

- Proxy-scale corpora only. No target-scale tokenizer has been trained and no
  tokenizer, shard, or report has been published.
- Fertility numbers are measurements of a named local split, not validated or
  comparable results, and are not a claim about other corpora or languages.
- Domain and language groups use the data engine's offline heuristic labels.
- The `uint16` shard format holds at most 65,536 IDs; a larger vocabulary would
  need a new format version.
