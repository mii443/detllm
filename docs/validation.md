# Validation Notes

This file records reproducible local checks for repository fixtures. It is not
a substitute for the external-model and cross-platform acceptance evidence in
`detllm-design.md`.

## Fixture Logits

Command:

```sh
cargo run -p det-cli -- logits -m testdata/tiny-f32.gguf --tokens "$(cat testdata/tiny.tokens.txt)" --hash --chunk-size 3
cargo run -p det-cli -- logits -m testdata/tiny-qmix.gguf --tokens "$(cat testdata/tiny.tokens.txt)" --hash --chunk-size 3
```

Observed hashes:

| fixture | hash |
|---|---|
| `testdata/tiny-f32.gguf` | `92a0280149c6b1505c84dce0d19486a2093f93b7978b579c220000d12e4ef7e7` |
| `testdata/tiny-qmix.gguf` | `8a34d3c4a05e9a30b90aadcdca7b6bac91655e6ab67980ccdb6726565d35f3e4` |

The `det-model` unit suite also checks these hashes across chunk sizes
`1`, `2`, `3`, and full length. With the `parallel` feature enabled, it checks
thread counts `1`, `2`, `7`, and `16`. The `det-cli` integration suite checks the
same fixture hashes through the public `logits --hash --chunk-size` CLI path.
It also verifies the tokenizer-backed prompt path by comparing
`detllm logits -p "abc\n" --hash --chunk-size 2` against the equivalent
explicit byte-token sequence `--tokens 97,98,99,10` for both fixtures.
The CLI suite also checks `logits --dump FILE`: the dumped little-endian f32
stream length must equal `positions * vocab * 4`, and hashing the dumped bytes
must reproduce the `--hash` stdout. This provides a stable artifact for
external raw-logits cosine-similarity sanity checks.
The model logits request boundary rejects token IDs outside the embedding
vocabulary before computing hashes or dump buffers, so raw `--tokens` input
cannot carry an out-of-vocabulary ID into the forward pass.

`det-model` also checks fixture position invariance directly: for each token
position, it compares logits from one continuous KV-cache run against logits
from replaying the prefix into a fresh cache. This runs for both `tiny-f32`
and `tiny-qmix`.

## Target-Model Raw Logits

The following local checks compare `detllm logits --dump` output against
llama.cpp C API logits from `scripts/reference_logits_llamacpp.cpp` using the
same tokenizer-backed prompt, `"Hello world from detllm validation."`. The
llama.cpp reference is run with `--sequential` so it matches detllm's
single-token KV-cache evaluation order; llama.cpp batched and sequential logits
can differ in later rows.

| model | tokens | vocab | overall cosine | min row cosine | status |
|---|---|---:|---:|---:|---|
| TinyLlama 1.1B Chat Q8_0 | `10994,3186,515,1439,645,112,8845,49` | 32000 | 0.999914931 | 0.999809620 | passes `--min-cosine 0.999` |
| Qwen2.5 1.5B Instruct Q8_0 | `9707,1879,504,3392,654,76,10519,13` | 151936 | 0.999840330 | 0.999647692 | passes `--min-cosine 0.999` |
| SmolLM2 1.7B Instruct Q8_0 | `19556,905,429,964,764,93,13132,30` | 49152 | 0.999467131 | 0.999227139 | passes `--min-cosine 0.999` |
| TinyLlama 1.1B Chat Q4_0 | `10994,3186,515,1439,645,112,8845,49` | 32000 | 0.999836229 | 0.999521848 | passes `--min-cosine 0.999` |

Representative command shape:

```sh
cargo run -p det-cli -- logits -m model.gguf --tokens TOKENS --hash --dump detllm.logits.bin
/tmp/reference_logits_llamacpp --model model.gguf --tokens TOKENS --out llama.logits.bin --expected-rows 8 --expected-vocab VOCAB --sequential --quiet
cargo run -p xtask -- compare-logits --actual detllm.logits.bin --reference llama.logits.bin --row-size VOCAB --rows 8 --min-cosine 0.999 --worst-rows 8 --top-diffs 5
```

The checked matrix can be reproduced with:

```sh
cargo build --release -p det-cli --features parallel,simd
scripts/run-target-logits-matrix.sh \
  --reference /tmp/reference_logits_llamacpp \
  --tinyllama-q8 /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  --tinyllama-q4 /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  --qwen25-q8 /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf \
  --smollm2-q8 /tmp/detllm-external/SmolLM2-1.7B-Instruct-Q8_0.gguf \
  --out /tmp/detllm-logits-matrix-rerun \
  --threads 8
```

Broader raw-logits matrix, using the same local host and reference binary:

```sh
scripts/run-target-logits-broad-matrix.sh \
  --reference /tmp/reference_logits_llamacpp \
  --tinyllama-q8 /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  --tinyllama-q4 /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  --qwen25-q8 /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf \
  --smollm2-q8 /tmp/detllm-external/SmolLM2-1.7B-Instruct-Q8_0.gguf \
  --out /tmp/detllm-logits-broad-matrix-20260710 \
  --threads 8
```

Observed broader matrix results:

| model | case | rows | vocab | overall cosine | min row cosine | status |
|---|---|---:|---:|---:|---:|---|
| TinyLlama Q8_0 | `ids-1-2-3` | 3 | 32000 | 0.999833181 | 0.999762988 | passes `--min-cosine 0.999` |
| TinyLlama Q8_0 | `hello-validation-8` | 8 | 32000 | 0.999914931 | 0.999809620 | passes `--min-cosine 0.999` |
| TinyLlama Q4_0 | `ids-1-2-3` | 3 | 32000 | 0.999667056 | 0.999624876 | passes `--min-cosine 0.999` |
| TinyLlama Q4_0 | `hello-validation-8` | 8 | 32000 | 0.999836229 | 0.999521848 | passes `--min-cosine 0.999` |
| Qwen2.5 Q8_0 | `special-hello-special` | 3 | 151936 | 0.999762170 | 0.999709642 | passes `--min-cosine 0.999` |
| Qwen2.5 Q8_0 | `hello-validation-8` | 8 | 151936 | 0.999840330 | 0.999647692 | passes `--min-cosine 0.999` |
| SmolLM2 Q8_0 | `ids-1-2-3` | 3 | 49152 | 0.999732621 | 0.999628577 | passes `--min-cosine 0.999` |
| SmolLM2 Q8_0 | `hello-validation-8` | 8 | 49152 | 0.999467131 | 0.999227139 | passes `--min-cosine 0.999` |

## Architecture Metadata Compatibility

`det-model` has a synthetic GGUF test for `general.architecture = "qwen2"`
using `qwen2.*` metadata keys with the same dense decoder tensor names. This
covers the first loader-level compatibility issue for Qwen2.5-family GGUFs
without claiming real-model correctness.
Qwen2-family attention projections may include optional
`blk.N.attn_{q,k,v}.bias` tensors. The loader reads those biases when present,
validates their projected vector lengths, and applies them after the Q/K/V
GEMV step before RoPE and KV-cache storage. Synthetic qwen2 coverage checks
that those optional tensors load, while llama-style fixtures continue to load
with no attention projection biases.

The loader also selects RoPE pairing by architecture, matching llama.cpp's
architecture mapping for the supported set: `llama` uses adjacent value pairs
and `qwen2` uses split-half/NeoX pairs. Unit tests cover both arithmetic
orders and verify that synthetic qwen2 metadata loads with split-half RoPE.
It also reads `{arch}.rope.dimension_count`, defaulting to the full attention
head dimension when the metadata is absent. A bit-level RoPE test covers
partial rotation and verifies that dimensions outside `rope.dimension_count`
remain unchanged for both adjacent and split-half pairing.
GGUF RoPE scaling metadata is not silently ignored: `{arch}.rope.scaling.type`
may be absent or `none`, while scaled variants such as `linear`, `yarn`, and
`longrope` are rejected until their deterministic table generation is
implemented.
The loader also reads optional `{arch}.attention.scale`; when absent, it uses
the design default `1 / sqrt(head_dim)`. Unit tests cover both the default and
metadata override paths, and the attention score helper has a bit-level check
for explicit scale application.
`LlamaConfig::from_gguf` validates the returned config before exposing it:
non-positive or non-finite RMS epsilon, non-positive or non-finite attention
scale, non-positive or non-finite RoPE base, and malformed RoPE dimension
counts are rejected at metadata load time instead of being deferred to the
first kernel call.
The implementation currently assumes key and value head lengths are both equal
to `embedding_length / attention.head_count`. Explicit
`{arch}.attention.key_length` and `{arch}.attention.value_length` metadata is
accepted only when it matches that value; differing lengths are rejected rather
than silently loading with the wrong KV-cache layout.
Model validation also requires the input token embedding rows and output
projection rows to match, so a hand-built `F32Llama` cannot expose different
input-token and logits vocabularies.
GGUF vocabulary size resolution treats malformed metadata as fatal rather than
as a missing value: wrong-typed `{arch}.vocab_size`, `llama.vocab_size`, or
`tokenizer.ggml.tokens` metadata returns `ModelError::Gguf` before falling back
to tensor dimensions.
It also validates hand-built model internals before execution: normalization
vectors and F32 weights must be finite, and quantized matrix block geometry
must match its declared row/column shape. The unit test
`validate_rejects_nonfinite_or_malformed_model_weights` covers those direct
`F32Llama::validate` boundaries.
Direct `F32Matrix` callers get the same boundary checks: public `row` and
`gemv` methods validate declared dimensions, backing storage length, and finite
weights before slicing or computing a dot product.
Public `WeightMatrix` methods apply the same checks before indexing quantized
block buffers. `gemv_and_residual_add_reject_nonfinite_outputs` covers
malformed quantized matrices returning `ModelError::Shape` instead of panicking
malformed direct F32 matrices returning `ModelError::Shape` or
`ModelError::NonFinite`, and quantized embedding row dequantization rejecting
non-finite outputs.
Likewise, metadata for decoder features that would change the implemented
forward equation is rejected unless it is explicitly neutral: parallel
residuals, non-causal attention, sliding-window attention, ALiBi bias, QKV
clamp, attention value scale, attention/final logit soft-capping, final logit
scale, embedding scale, residual scale, and output embedding lengths that differ
from the hidden embedding length. These guards keep real GGUFs with unsupported
architecture features from producing plausible but incorrect deterministic
logits. Nonzero `F64` feature metadata is rejected using the original metadata
value before any f32 rounding, so tiny nonzero values cannot underflow to
`0.0` and pass as neutral.

## Dense Tensor Type Compatibility

`det-model` accepts GGUF dense `F32` and `F16` tensors for embeddings,
normalization vectors, projections, and output weights. F16 values are expanded
with the deterministic bit-level converter from `det-num`, and non-finite F16
values are rejected at load time. Synthetic model tests cover both the F16
forward path and non-finite rejection.
`det-gguf` also checks tensor encoded byte-length arithmetic all the way through
the final block-count times type-size multiplication, so impossible tensor
shapes fail with `InvalidTensorShape` rather than wrapping the byte length.
The parser also rejects duplicate metadata keys and duplicate tensor names
instead of silently choosing a first or last definition, keeping validation and
later tensor lookup aligned on one unambiguous GGUF view.
The public tensor lookup path keeps the same invariant for caller-constructed
`Gguf::from_parts` values: duplicate tensor names return
`DuplicateTensorName` instead of selecting the first match.
Boolean metadata values are canonicalized at the parser boundary: scalar and
array bool entries must be encoded as byte `0` or `1`, and any other byte is
reported as malformed instead of being treated as truthy.
Tensor payload ranges are validated during parse as well: encoded lengths are
checked against the file length, and overlapping tensor data ranges are
rejected as `InvalidTensorOffset` before any later tensor lookup can observe an
ambiguous byte region.
`det-model` mirrors that policy at the model boundary: F32 matrix construction,
dense tensor byte-length checks, quantized block counts, and public GGUF GEMV
shape checks use checked `usize` arithmetic and return `ModelError::Shape` on
overflow.
GGUF `Q4_K` weight matrices are accepted alongside `Q8_0`, `Q4_0`, and `Q6_K`.
The scalar Q4_K path follows the GGML 256-value block layout (`d`, `dmin`,
12-byte scale/min table, and 128 packed quant bytes), and unit tests cover the
scale/min bit unpacking, Q8A dot path, row dequantization, shared-Q8A GEMV, and
parallel row partitioning invariance.

## RoPE Kernel Order

`det-model` has a bit-level unit test for RoPE application over multiple heads.
It fixes the current adjacent-pair rotation order
`x[2j], x[2j + 1] -> (x0 * c - x1 * s, x0 * s + x1 * c)` so future changes do
not accidentally alter the deterministic arithmetic sequence before real-model
cosine checks are in place.

RoPE table lookup and application also reject malformed tables, non-finite
table entries, non-finite inputs, and non-finite rotated outputs. The unit test
`rope_rejects_malformed_or_nonfinite_values` covers those DET-4 paths.
The public `forward_one` boundary also requires caller-provided `RopeTables`
to use the same pairing as the model config and caller-provided `KvCache` to
carry the same config as the model. The unit test
`forward_one_rejects_mismatched_rope_or_cache_config` covers those state
consistency checks.
`det-model` also exposes `ForwardWorkspace` so repeated token evaluation can
reuse the large temporary hidden-state, projection, attention, and feed-forward
buffers instead of allocating them for every `forward_one` call. The ordinary
`forward_one` API is retained as a compatibility wrapper, while logits hashing,
CLI compression, and `xtask bench-file` use the workspace path. The unit test
`forward_one_workspace_matches_default_forward` verifies that the reusable
workspace path produces the same logits bits as the default wrapper across a
multi-token KV-cache run.
Attention now reads contiguous KV-cache prefixes directly instead of copying
the key/value window into scratch buffers for every layer/head. With the
`parallel` feature, attention scratch is split per head so independent heads
can run in parallel for larger attention windows while each head keeps its
score softmax and value accumulation order. The unit test
`kv_cache_prefix_slices_are_contiguous_and_bounds_checked` covers the direct
prefix-slice layout and rejects malformed prefix requests.
Size arithmetic for RoPE tables and logits dumps is checked before allocation.
Overflowing table lengths, matrix lengths, or logits byte lengths return
`ModelError::Shape` instead of panicking. KV-cache allocation and public
key/value slice lookup use the same checked arithmetic policy, and the logits
dump path validates token-window arguments before computing its output capacity.

## Kernel Non-Finite Rejection

`det-model` public kernels reject non-finite inputs and non-finite outputs with
`ModelError::NonFinite` instead of letting NaN-dependent values propagate. The
unit test `public_kernels_reject_nonfinite_inputs` covers RMSNorm, attention
score calculation, attention weighted-value accumulation, SwiGLU, and softmax
error paths. The main `forward_one` path uses the public attention score and
weighted-value kernels, and fixture logits/position-invariance tests cover that
refactoring without changing golden hashes.
Public kernel shape checks also use checked length arithmetic before comparing
slice sizes. `public_kernels_reject_shape_overflow_or_empty_attention` covers
overflowing RoPE shape products plus empty attention score/value calls returning
`ModelError::Shape` instead of panicking or accepting a meaningless no-op.

GEMV and residual-addition paths also reject non-finite inputs or non-finite
outputs. The unit test `gemv_and_residual_add_reject_nonfinite_outputs` covers
F32 GEMV input rejection, F32 GEMV overflow detection, quantized GEMV input
rejection, residual-add overflow detection, and non-finite F32 matrix
construction. `kv_cache_store_rejects_nonfinite_values` covers the public KV
cache store boundary, while
`kv_cache_rejects_out_of_bounds_indices_and_bad_lengths` covers malformed KV
payload lengths plus out-of-bounds layer, KV-head, and position reads/writes.
The public GGUF `F32TensorView` GEMV helper is covered by
`f32_gemv_from_view_rejects_nonfinite_boundaries`, which checks normal output,
non-finite input rejection, and non-finite output rejection.

## Tokenizer Losslessness

`det-token` unit tests round-trip all byte values `0x00..0xff` through the
byte fallback tokenizer, GPT-2 byte-unicode BPE tokenizer, and SentencePiece
style tokenizer. The BPE and SentencePiece tests also include merged/piece
tokens in the vocabulary, so the checks cover both byte fallback and multi-byte
token detokenization paths.

CLI paths that pair the tokenizer with the model reject GGUFs where
`tokenizer.ggml.tokens` has a different length from the model output
vocabulary. This keeps compression, decompression, and prompt-backed logits
from carrying a tokenizer/model vocabulary mismatch into CDF construction,
detokenization, or logits dumps.
Codec paths reserve 256 deterministic byte escape symbols after the model
vocabulary. If a tokenizer cannot emit a byte as a vocabulary token, compression
encodes `vocab_len + byte`; decompression writes that byte directly and does
not advance the model context for the escape symbol. The CDF assigns these
escape symbols minimum frequency after normal logits-derived vocabulary
symbols, while the initial empty-context CDF is uniform over vocabulary plus
escapes. New DTLZ files set `FLAG_BYTE_ESCAPES` in the header; `flags=0`
remains the legacy token-only CDF mode. Unit tests cover partial-BPE inputs
where present bytes still use BPE tokens and missing bytes use byte escapes.
The byte-token mapping must also be unambiguous. `det-token` rejects duplicate
canonical `<0xNN>` byte fallback entries, and BPE tokenizers reject duplicate
emittable token byte sequences, including single-byte tokens, instead of
allowing a later token ID to overwrite the byte-to-token or merge-target
mapping. The public byte-fallback constructor enforces the same invariant for
caller-provided tables: neither a byte nor a token ID may appear twice.

Tokenizer metadata validation also rejects malformed `tokenizer.ggml.scores`
arrays. The scores array must match the token vocabulary length and contain
only finite `f32` values, preventing NaN/Inf score comparisons from entering
the SentencePiece selection path. The direct SentencePiece constructor shares
the same validation, so callers cannot bypass this check with in-memory token
tables.
`tokenizer.ggml.token_type` metadata must also use known GGUF token type values:
normal, unknown, control, user-defined, unused, or byte. Unknown numeric values
are rejected instead of being silently treated as non-emittable.
Non-emittable token types are excluded from compression token selection while
remaining decodable if they appear in an existing stream. Byte fallback
tokenizers apply the same emit mask: present but non-emittable byte fallback
tokens are not emitted by new codec streams, and those bytes use deterministic
byte escape symbols instead. The unit tests
`byte_fallback_codec_escapes_nonemittable_byte_tokens_from_token_type` and
`sentencepiece_codec_escapes_nonemittable_byte_fallback_tokens` cover that
boundary.
GPT-2/BPE merge metadata is also parsed strictly: each merge line must contain
exactly two non-empty token pieces, and duplicate merge pairs are rejected
instead of letting a later entry overwrite the earlier rank. This keeps BPE
rank ordering a single unambiguous function of the GGUF metadata.
The ByteBPE tokenizer applies merges with a priority queue over linked token
nodes so large inputs do not require a full rank scan plus `Vec::remove` for
every merge. The test `byte_bpe_priority_queue_matches_rank_scan_reference`
checks the optimized path against the direct rank-scan rule on overlapping
merge patterns, preserving the rank-then-leftmost tie semantics.
Tokenizer model metadata is not silently ignored: `tokenizer.ggml.model = gpt2`
selects the ByteBPE path even when the merge list is empty, SentencePiece model
names select the SentencePiece path, and unknown model names or wrong metadata
types are rejected instead of falling back to byte fallback tokenization.

## Quant Kernel Canary

Command:

```sh
cargo run -p det-cli -- quant-kernel-hash
```

Observed hash:

```text
99832eb2ac8ddeb15731805e876a36b4013ae41c2aca0783ea02890fe9b0efba
```

`det-quant` rejects non-finite Q8A activation inputs before quantization.
The unit test `q8a_rejects_nonfinite_and_bad_block_lengths` covers NaN, Inf,
zero-length activation vectors, and invalid block lengths, and `det-model` maps this path to
`ModelError::NonFinite`. This is part of the DET-4 defense against silently
creating NaN-dependent outputs.
The public quantized dot APIs also reject non-finite weight or activation
scales on caller-constructed blocks before dispatching to scalar or SIMD block
kernels. They also reject non-finite block outputs or block-sequential sums
that can arise from finite but overflowing caller-provided scales.
Empty quantized block lists are rejected as invalid lengths rather than
treated as a valid zero dot product.
The public F32 row-major GEMV helper now reports shape mismatches, non-finite
inputs, and non-finite outputs through `QuantError` instead of panicking or
returning an unchecked result.
`det-model` maps those quantized-dot failures to `ModelError::NonFinite`, and
its GEMV boundary test covers finite Q8/Q4 inputs that overflow during the
quantized dot path.
The CLI startup runtime canary also hashes a fixed set of
Q8_0/Q4_0/Q4_K/Q6_K block-dot outputs before executing normal commands, so a
broken selected quantized dot backend is caught by `selftest` and by ordinary
CLI entry points, not only by the separate `quant-kernel-hash` diagnostic
command. `quant-kernel-hash` itself covers 1,000,000 deterministic Q8_0/Q4_0
block cases plus 4,096 deterministic Q4_K block cases and 4,096 deterministic
Q6_K block cases, and local scalar and AVX2 SIMD runs produced the same hash
shown above.
The `shared_q8a_path_matches_standalone_quantized_gemv` test fixes the
`detllm-design.md` §5.2 quantization timing rule: one Q8A activation buffer is
created for mixed F32/quantized projection groups when any matrix needs it,
quantized GEMV requires that shared buffer, and the shared-buffer results match
standalone quantized GEMV bit-for-bit.
The `quant_scratch_gemv_path_matches_standalone_quantized_gemv` test covers the
forward-path scratch APIs directly: Q8_0, Q4_0, and Q4_K reuse caller-provided
Q8A buffers while Q6_K fills Q8_K scratch, and every scratch result matches the
standalone quantized GEMV bits.

Local validation after the bench checkpoint/resume changes on commit
`8e0b756921caf6c568af20543ea6ae0dcb00f1b1`:

```sh
cargo test --workspace
cargo test --workspace --features parallel,simd
cargo run -p xtask -- check-determinism
cargo run -p xtask -- check-ci-workflow
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --all-targets --features parallel,simd -- -D warnings
```

All six commands passed locally on 2026-07-10. The `parallel,simd` test run
covered the full workspace, including `parallel_gemv_thread_counts_are_bit_invariant`,
`testdata_logits_hash_is_invariant_to_chunks_and_threads`, and the
`bench_file_checkpoint_resume_matches_one_shot_payload` checkpoint regression.
Additional local CLI smoke on 2026-07-10 after commit
`629014ecf0c786b84c7fd8176b0582b64a0fe8cc`:

```sh
cargo run -p xtask -- generate-testdata --check
cargo run -p det-cli -- selftest
cargo run -p det-cli -- quant-kernel-hash
cargo run -p det-cli -- logits -m testdata/tiny-f32.gguf --tokens 0,1,2,3,0,2 --hash --chunk-size 3
cargo run -p det-cli -- logits -m testdata/tiny-qmix.gguf --tokens 0,1,2,3,0,2 --hash --chunk-size 3
```

All five commands passed locally. The observed CLI hashes were:

| check | hash |
|---|---|
| `quant-kernel-hash` | `99832eb2ac8ddeb15731805e876a36b4013ae41c2aca0783ea02890fe9b0efba` |
| `tiny-f32` logits | `92a0280149c6b1505c84dce0d19486a2093f93b7978b579c220000d12e4ef7e7` |
| `tiny-qmix` logits | `8a34d3c4a05e9a30b90aadcdca7b6bac91655e6ab67980ccdb6726565d35f3e4` |

`quant-kernel-hash` also returned the same
`99832eb2ac8ddeb15731805e876a36b4013ae41c2aca0783ea02890fe9b0efba` value in
release mode and with `--features parallel,simd`.

GitHub Actions `ci` also passed for commit
`3dd1e0903682544a052b812294933a67c6eba449`:
<https://github.com/mii443/detllm/actions/runs/29072369651>. The public
Actions API reported `status=completed`, `conclusion=success`, run number 123,
created `2026-07-10T05:52:00Z`, updated `2026-07-10T05:53:11Z`.

The AVX2 SIMD kernel path is also executed directly in CI with:

```sh
RUSTFLAGS="-C target-feature=+avx2" cargo test -p det-quant --features simd simd_blocks_match_scalar_bits
```

Local run on `x86_64` passed this test over 1,000,000 deterministic block
cases, comparing Q8_0/Q4_0 SIMD block dots against the scalar implementation
by exact `f32::to_bits()` equality.

## Determinism Hygiene

Command:

```sh
cargo run -p xtask -- check-determinism
```

The check scans implementation and CI files for `detllm-design.md` banned
constructs such as platform transcendental calls, `mul_add`, randomized
`HashMap`/`HashSet` usage, wasm `relaxed-simd`, and Rayon parallel reductions,
including reductions hidden behind iterator adaptors such as `.map(...)`.
Host-dependent `target-cpu=native` builds are rejected as well; target features
used for validation must be explicit and covered by bit-equivalence tests.
FMA target features are rejected whether they appear as a direct
`target-feature=+fma` flag or inside a feature list such as
`target-feature=+avx2,+fma` or `#[target_feature(enable = "avx2,fma")]`.
Inline assembly via `asm!` or `global_asm!` is rejected because it can bypass
the reviewed `core::arch` SIMD paths and their bit-equivalence tests.
Explicit floating-point iterator reductions such as `.sum::<f32>()`,
`Iterator::sum::<f32>(...)`, `.product::<f32>()`, inferred `.sum()` or
`.product()` calls, `.reduce(...)`, `.fold(...)`, `.try_fold(...)`, and the
corresponding UFCS forms are also rejected; numeric reductions must use the
fixed 8-lane helpers or the locally specified sequential accumulation sites.
Integer-only or otherwise nonnumeric iterator reductions require a local
`determinism-allow` marker so future inferred floating-point reductions cannot
hide behind assignment-side type inference.
It covers both associated-function spellings such as `f64::exp` and method-call
spellings such as `x.exp()`, so validation helpers cannot accidentally
reintroduce platform libm through Rust's primitive float methods.
The platform-libm guard also covers adjacent primitive float transcendental
helpers such as arbitrary-base `log`, `ln_1p`, `exp_m1`, `exp2`, `log2`,
`log10`, `powi`, `sin_cos`, `tan`, inverse trig, hyperbolic trig including
`asinh`/`acosh`/`atanh`, `cbrt`, and `hypot`; `sqrt` remains allowed by
`detllm-design.md` because IEEE 754 requires correctly rounded square root.
It intentionally excludes prose docs and the design file itself to avoid
flagging normative descriptions. The GitHub Actions `hygiene` job runs this
check after stale-testdata validation.
For `Cargo.toml` files, the same check also enforces dependency hygiene: path
dependencies are accepted, while external dependencies must use exact
`=x.y.z` versions. This keeps future third-party additions aligned with the
`detllm-design.md` requirement that numerically relevant dependencies not float
across builds.
Native numeric dependencies and wrappers such as BLAS/LAPACK/OpenBLAS/MKL,
SLEEF, and external `libm` crates are rejected alongside native build/link
tooling; deterministic math must stay in reviewed Rust source or vendored code.
The dependency hygiene check also inspects `Cargo.lock` package names for the
same native build/link tooling list, so a seemingly harmless direct dependency
cannot hide a transitive `cc`, `cmake`, BLAS, SLEEF, or external `libm` package.

## Compression Smoke

Input:

```text
detllm deterministic compression smoke
```

Command:

```sh
tmpdir="$(mktemp -d /tmp/detllm-readme.XXXXXX)"
printf 'detllm deterministic compression smoke\n' > "$tmpdir/input.txt"
cargo run -p det-cli -- compress -m testdata/tiny-f32.gguf -i "$tmpdir/input.txt" -o "$tmpdir/out.dtlz" --n-ctx 8 --threads 3
cargo run -p det-cli -- decompress -m testdata/tiny-f32.gguf -i "$tmpdir/out.dtlz" -o "$tmpdir/restored.txt" --threads 3
cmp "$tmpdir/input.txt" "$tmpdir/restored.txt"
wc -c "$tmpdir/input.txt" "$tmpdir/out.dtlz"
```

Observed size:

| input bytes | compressed bytes | note |
|---:|---:|---|
| 39 | 102 | includes the 56-byte DTLZ header; tiny fixture is for round-trip validation, not compression quality |

The `det-cli` integration suite also checks DTLZ payload canonicality. After
decompression, the CLI re-encodes the restored token stream with the same
model, context, and overlap settings, then rejects the file if the payload does
not match byte-for-byte. The test
`decompress_rejects_noncanonical_payload_with_trailing_bytes` verifies that a
valid compressed file with extra trailing payload bytes is rejected instead of
silently restoring the prefix.
The lower-level `det-coder::decode_token_stream` API now performs the same
canonicality check for caller-provided CDF streams by re-encoding the decoded
tokens and requiring byte-for-byte equality with the input payload. The unit
test `token_stream_rejects_noncanonical_trailing_payload` covers this stream
boundary directly.
The unit test `cli_decompress_truncates_final_multibyte_token_to_orig_len`
covers the file-format rule that `orig_len` is authoritative even when the
last decoded token emits more than one byte: the decompressor writes only the
first `orig_len` restored bytes.
Compression also rejects `--n-ctx` values larger than the loaded model's
declared context length instead of silently clamping them before writing the
DTLZ header. This keeps the recorded window settings equal to the caller's
explicit request.
DTLZ header invariants are checked on both decode and checked-encode paths:
unknown flag bits, zero `n_ctx`, and `overlap >= n_ctx` are rejected before a
header is accepted or written by the CLI. `FLAG_BYTE_ESCAPES` is accepted and
is written by new CLI compression output; `flags=0` remains accepted for
legacy token-only payloads.
The public CLI integration round-trip over both repository GGUF fixtures also
decodes each produced DTLZ header and checks the written byte-escape flag,
requested `n_ctx`, derived `overlap`, `orig_len`, and model SHA-256 against the
actual compressed input and model bytes.
The unit test `rejects_malformed_header_envelope` also covers too-short files,
bad magic bytes, and unsupported header versions before any payload decoding is
attempted.
The CLI unit test `cli_decompress_rejects_malformed_header_before_model_load`
fixes the decompression order: malformed DTLZ headers are rejected before the
model path is loaded, so bad compressed input cannot trigger unnecessary model
parsing or output creation.
Decompression rejects `--n-ctx` overrides because `n_ctx` and `overlap` are
part of the DTLZ header and must be replayed exactly from the compressed file.
The unit test `model_backed_token_codec_rejects_invalid_windows` covers the
lower-level model-backed encode/decode helpers directly: both sides reject
zero `n_ctx`, `overlap >= n_ctx`, and `n_ctx` larger than the model context
before any token stream is processed.
The decompressor also avoids preallocating the full header `orig_len` before
range decoding. A corrupted file with an inflated `orig_len` is expected to
fail without writing the restored output path, rather than attempting a huge
allocation from untrusted file metadata.

`det-coder` validates public `Cdf` values before stream encoding or decoding.
The tests `cdf_validate_rejects_malformed_tables` and
`token_stream_rejects_malformed_cdf_without_panicking` cover empty tables,
length mismatches, zero frequencies, bad prefix sums, and total mismatches, so
malformed caller-supplied CDFs are reported as errors instead of indexing into
invalid tables. `cdf_validate_rejects_malformed_tables` also fixes the
range-coder precondition that public CDF totals must remain below `2^31`.
The public `Cdf::symbol_for` helper performs the same validation before binary
searching the prefix table; `decode_token_stream` uses an internal
already-validated helper to avoid repeating that O(V) check for each symbol.
`rejects_invalid_frequency_ranges_without_overflow` also covers corrupted
range-coder payloads whose current code value would decode to a frequency
outside `0..total`; `RangeDecoder::decode_freq` returns `InvalidFrequency`
instead of exposing an out-of-range symbol position.
The decoder also enforces the public API protocol: a successful `decode_freq`
must be followed by exactly one `advance` using the same total before another
frequency can be decoded. `decoder_enforces_decode_advance_pairs` covers
double-decode, advance-without-decode, and mismatched-total misuse returning
`InvalidFrequency` instead of corrupting decoder state.
The integration test `range_coder_round_trips_large_lcg_stream` covers the
`detllm-design.md` §8.1 large-stream acceptance path with a deterministic LCG
frequency table and 1,000,000 encoded symbols, then decodes the whole stream
symbol-for-symbol through the public range-coder API.
The stream-level test `token_stream_rejects_corrupted_payload_frequency` checks
that the same corrupted-payload condition is surfaced through
`decode_token_stream` as `StreamError::Range`.
The unit test `logits_to_cdf_does_not_redistribute_to_fixed_total` covers the
`detllm-design.md` §5.6 rule that logits-derived frequencies are not
redistributed back to exactly `2^24`; minimum-frequency increments are kept in
the range-coder total.
The CDF builder also enforces the design vocabulary bound of `2^18` symbols
before allocating its intermediate softmax vectors, and public `Cdf::validate`
applies the same bound to caller-constructed tables. The unit suite covers the
accepted boundary and the first rejected symbol count for logits-derived,
uniform, and public CDF validation paths. The CLI `LoadedModel` path applies
the same bound before a model is accepted for compression or decompression, so
codec use fails at model load time instead of after the first CDF construction.
With the `parallel` feature, CDF construction fills the independent
`exp_f32(logit[i] - max)` scratch slots with Rayon, then returns to the
normative single-threaded 8-lane `Z` reduction and prefix-sum construction.
The unit test `logits_to_cdf_matches_scalar_reference_for_large_vocab`
compares the feature-selected path against a scalar reference over a 10,003
symbol deterministic vector.

## Local Testdata Bench Harness

Command:

```sh
cargo run --release -p xtask -- bench-testdata --iters 100
```

The command measures the repository fixtures only:

- `logits tiny-f32`
- `logits tiny-qmix`
- `codec tiny-f32`
- `codec tiny-qmix`

Each measurement verifies the fixture hash or codec round-trip on every
iteration before reporting elapsed time and throughput. The values are useful
for local regressions, but they are not the real-model benchmark evidence
required by `detllm-design.md`.

## External Model Intake Harness

Command:

```sh
cargo run -p xtask -- model-info --model testdata/tiny-f32.gguf
cargo run -p xtask -- model-info --model model.gguf
cargo run -p xtask -- model-info --model model-prefix.gguf --metadata-prefix
```

`model-info` is the lightweight first step for TinyLlama / SmolLM2 / Qwen2.5
GGUF validation. It parses the file, reports the model SHA-256, selected GGUF
metadata, tokenizer kind, byte coverage, deterministic `LlamaConfig`
interpretation, tensor type inventory, tokenizer/model/codec vocabulary
compatibility, and the required tensor shape/type status. It does not
instantiate `F32Llama`, so it can be run before a full logits or compression
pass on larger external GGUFs.
With `--metadata-prefix`, the same summaries can be run on a GGUF prefix that
contains the header, metadata, and tensor table but not tensor payload bytes.
This is useful for screening external GGUF candidates for tokenizer byte
coverage and tensor shape/type compatibility before downloading multi-GB model
files; payload-backed commands such as `logits` and `bench-file` still require
the complete GGUF.

Observed smoke output on the bundled F32 fixture:

```text
model-info path=testdata/tiny-f32.gguf bytes=13520 sha256=ce2aa01900a63585a409ef995a2827dcac81e1678e38a1ab0733302ba82ce79b metadata_prefix=false gguf_version=3 metadata=12 tensors=12 data_offset=4800
model-info metadata key=general.architecture string=llama
model-info metadata key=llama.vocab_size u32=256
model-info metadata key=tokenizer.ggml.tokens array<string>[256]
model-info tokenizer status=ok kind=byte_fallback
model-info byte-coverage tokens=256 single_byte=256 emittable_single_byte=256 missing=0 missing_emittable=0 missing_first=none missing_emittable_first=none
model-info config status=ok block_count=1 embedding_length=4 feed_forward_length=6 head_count=2 head_count_kv=1 context_length=16 rope_dimension_count=2 rope_pairing=Adjacent rope_freq_base=10000.0 rms_epsilon=1e-5 attention_scale=0.70710677
model-info tensor-inventory total=12 encoded_bytes=8720 encoded_len_errors=0 F32=12
model-info vocab status=ok tokenizer=256 model=256 codec_max_symbols=262144
model-info required-tensors status=ok checked=12 missing=0 shape_mismatch=0 unsupported_type=0 tied_output=false
```

For external validation, record this output next to the logits hash, cosine
comparison, and `bench-file` output. The required tensor status is evidence
that the target model has the tensor names, dimensions, and tensor types that
the deterministic inference path will attempt to load.

### TinyLlama External Smoke

Source:

- Repository: <https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF>
- Supported model file:
  `tinyllama-1.1b-chat-v1.0.Q8_0.gguf`
- Additional supported model file:
  `tinyllama-1.1b-chat-v1.0.Q4_0.gguf`

The Q4_0 file contains one `Q6_K` tensor for `output.weight`, so it exercises
both the corrected GGML Q4_0 low-half/high-half nibble order and the scalar
Q6_K output projection path with Q8_K activation quantization.

Observed Q4_0 intake result:

```text
model-info path=/tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q4_0.gguf bytes=637699456 sha256=da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556 metadata_prefix=false gguf_version=3 metadata=23 tensors=201 data_offset=1709440
model-info tokenizer status=ok kind=sentencepiece
model-info byte-coverage tokens=32000 single_byte=488 emittable_single_byte=488 missing=0 missing_emittable=0 missing_first=none missing_emittable_first=none
model-info config status=ok block_count=22 embedding_length=2048 feed_forward_length=5632 head_count=32 head_count_kv=4 context_length=2048 rope_dimension_count=64 rope_pairing=Adjacent rope_freq_base=10000.0 rms_epsilon=1e-5 attention_scale=0.125
model-info tensor-inventory total=201 encoded_bytes=635990016 encoded_len_errors=0 F32=45 Q4_0=155 Q6_K=1
model-info vocab status=ok tokenizer=32000 model=32000 codec_max_symbols=262144
model-info required-tensors status=ok checked=201 missing=0 shape_mismatch=0 unsupported_type=0 tied_output=false
```

Observed Q4_0 logits smoke:

```sh
cargo run --release -p det-cli -- logits -m /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --tokens 1 --hash --threads 8
cargo run --release -p det-cli -- logits -m /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --tokens 1,2,3 --hash --chunk-size 1 --threads 8
cargo run --release -p det-cli -- logits -m /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --tokens 1,2,3 --hash --chunk-size 3 --threads 8
```

Observed output:

```text
tokens=1 hash = e3908cc604210dac0ad8c31543c40eaebc34862e9e7cdcb38d2503b44a3944b0
tokens=1,2,3 chunk-size=1 hash = 450bf34ee63249f042cde2156643a53261034a4fa04bf47721da9d865ada9251
tokens=1,2,3 chunk-size=3 hash = 450bf34ee63249f042cde2156643a53261034a4fa04bf47721da9d865ada9251
```

Q4_0 raw logits llama.cpp reference:

```sh
cargo run --release -p det-cli -- logits -m /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --tokens 1,2,3 --dump /tmp/detllm-tinyllama-q4-123.rawlogits.bin --hash --threads 8
/tmp/reference_logits_llamacpp --model /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --tokens 1,2,3 --out /tmp/llamacpp-tinyllama-q4-123.rawlogits.bin --threads 8 --ctx-size 16 --batch-size 16 --expected-vocab 32000 --expected-rows 3 --quiet
cargo run -p xtask -- compare-logits --actual /tmp/detllm-tinyllama-q4-123.rawlogits.bin --reference /tmp/llamacpp-tinyllama-q4-123.rawlogits.bin --row-size 32000 --rows 3 --min-cosine 0.999
```

Observed output:

```text
450bf34ee63249f042cde2156643a53261034a4fa04bf47721da9d865ada9251
reference_logits_llamacpp rows=3 vocab=32000 values=96000
compare-logits values=96000 cosine=0.999667056 max_abs_diff=0.416365802 rms_diff=0.064927172 rows=3 row_size=32000 min_row=2 min_row_cosine=0.999624876
```

Observed Unsloth Q8_0 intake result:

```text
model-info path=/tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q8_0.gguf bytes=1170781568 sha256=a4c9bb1dbaa372f6381a035fa5c02ef087aaa1ff1f843a56a22328114f03fc59 metadata_prefix=false gguf_version=3 metadata=23 tensors=201 data_offset=1709440
model-info metadata key=general.architecture string=llama
model-info metadata key=general.name string=tinyllama_tinyllama-1.1b-chat-v1.0
model-info metadata key=tokenizer.ggml.model string=llama
model-info metadata key=tokenizer.ggml.bos_token_id u32=1
model-info metadata key=tokenizer.ggml.eos_token_id u32=2
model-info metadata key=tokenizer.ggml.tokens array<string>[32000]
model-info metadata key=tokenizer.ggml.merges array<string>[61249]
model-info metadata key=tokenizer.ggml.scores array<f32>[32000]
model-info metadata key=tokenizer.ggml.token_type array<i32>[32000]
model-info tokenizer status=ok kind=sentencepiece
model-info byte-coverage tokens=32000 single_byte=488 emittable_single_byte=488 missing=0 missing_emittable=0 missing_first=none missing_emittable_first=none
model-info config status=ok block_count=22 embedding_length=2048 feed_forward_length=5632 head_count=32 head_count_kv=4 context_length=2048 rope_dimension_count=64 rope_pairing=Adjacent rope_freq_base=10000.0 rms_epsilon=1e-5 attention_scale=0.125
model-info tensor-inventory total=201 encoded_bytes=1169072128 encoded_len_errors=0 F32=45 Q8_0=156
model-info vocab status=ok tokenizer=32000 model=32000 codec_max_symbols=262144
model-info required-tensors status=ok checked=201 missing=0 shape_mismatch=0 unsupported_type=0 tied_output=false
```

Minimal logits smoke:

```sh
cargo run --release -p det-cli -- logits -m /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q8_0.gguf --tokens 1 --hash --threads 8
cargo run --release -p det-cli -- tokenize -m /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q8_0.gguf -p "Hello"
cargo run --release -p det-cli -- logits -m /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q8_0.gguf --tokens 1,2,3 --hash --chunk-size 1 --threads 8
cargo run --release -p det-cli -- logits -m /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q8_0.gguf --tokens 1,2,3 --hash --chunk-size 3 --threads 8
```

Observed output:

```text
tokens("Hello") = 10994
tokens=1 hash = 6e485ce2165e7c50da0297576fa56a3528f79ebf0fca0f25a160b61331543248
tokens=1,2,3 chunk-size=1 hash = 79600ae16f6ba067de254839a0df605a1082b2eb6f75b538411be9403fe9251c
tokens=1,2,3 chunk-size=3 hash = 79600ae16f6ba067de254839a0df605a1082b2eb6f75b538411be9403fe9251c
```

Minimal codec smoke:

```sh
printf 'Hello\n' > /tmp/detllm-external/hello.txt
cargo run --release -p xtask -- bench-file --model /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q8_0.gguf --input /tmp/detllm-external/hello.txt --n-ctx 16 --iters 1
```

Observed output:

```text
bench-file model=/tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q8_0.gguf input=/tmp/detllm-external/hello.txt limit_bytes=all limit_tokens=all iters=1 warmup=true threads=default n_ctx=16 overlap=4 model_sha256=a4c9bb1dbaa372f6381a035fa5c02ef087aaa1ff1f843a56a22328114f03fc59 input_sha256=66a045b452102c59d840ec097d59d9467e13a3f34f6494e539ffd32c1bb35f18
bench-file: source_input_bytes=6 measured_input_bytes=6 total_input_bytes=6 tokens=2 total_tokens=2 payload_bytes=10 dtlz_bytes=66 payload_bits_per_byte=13.333333 dtlz_bits_per_byte=88.000000 compression_ratio=11.000000 elapsed_ms=344.482 input_bytes_per_s=17.417 tokens_per_s=5.806
```

This is real TinyLlama GGUF evidence for tokenizer construction, model config
parsing, full required tensor compatibility on Q8_0, single-token forward,
chunk-size-invariant logits hashing on a three-token stream, and an end-to-end
codec round-trip on a tiny byte input. It is not a substitute for the HF
transformers raw-logits cosine check or the enwik8 first-1MB compression
measurement; the llama.cpp perplexity-path log-probability check is recorded
below.

### Qwen2.5 External Smoke

Source:

- Repository: <https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF>
- Supported model file:
  `qwen2.5-1.5b-instruct-q8_0.gguf`

Observed Q8_0 intake result:

```text
model-info path=/tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf bytes=1894532128 sha256=d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8 metadata_prefix=false gguf_version=3 metadata=26 tensors=339 data_offset=5950496
model-info metadata key=general.architecture string=qwen2
model-info metadata key=general.name string=qwen2.5-1.5b-instruct
model-info metadata key=tokenizer.ggml.model string=gpt2
model-info metadata key=tokenizer.ggml.add_bos_token bool=false
model-info metadata key=tokenizer.ggml.bos_token_id u32=151643
model-info metadata key=tokenizer.ggml.eos_token_id u32=151645
model-info metadata key=tokenizer.ggml.tokens array<string>[151936]
model-info metadata key=tokenizer.ggml.merges array<string>[151387]
model-info metadata key=tokenizer.ggml.token_type array<i32>[151936]
model-info tokenizer status=ok kind=byte_bpe
model-info byte-coverage tokens=151936 single_byte=256 emittable_single_byte=256 missing=0 missing_emittable=0 missing_first=none missing_emittable_first=none
model-info config status=ok block_count=28 embedding_length=1536 feed_forward_length=8960 head_count=12 head_count_kv=2 context_length=32768 rope_dimension_count=128 rope_pairing=SplitHalf rope_freq_base=1000000.0 rms_epsilon=1e-6 attention_scale=0.088388346
model-info tensor-inventory total=339 encoded_bytes=1888581632 encoded_len_errors=0 F32=141 Q8_0=198
model-info vocab status=ok tokenizer=151936 model=151936 codec_max_symbols=262144
model-info required-tensors status=ok checked=255 missing=0 shape_mismatch=0 unsupported_type=0 tied_output=false
```

Minimal logits smoke:

```sh
cargo run --release -p det-cli -- tokenize -m /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf -p "Hello"
cargo run --release -p det-cli -- logits -m /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf --tokens 151643 --hash --threads 8
cargo run --release -p det-cli -- logits -m /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf --tokens 151643,9707,151645 --hash --chunk-size 1 --threads 8
cargo run --release -p det-cli -- logits -m /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf --tokens 151643,9707,151645 --hash --chunk-size 3 --threads 8
```

Observed output:

```text
tokens("Hello") = 9707
tokens=151643 hash = 26c0784ba271b6a72170625ff878536576b6a1618a65db476721ce562f1ccba6
tokens=151643,9707,151645 chunk-size=1 hash = 54a2d926c73430df3b849f91ad106f0723bc5f007d851c2767e8b115e1ca7fc7
tokens=151643,9707,151645 chunk-size=3 hash = 54a2d926c73430df3b849f91ad106f0723bc5f007d851c2767e8b115e1ca7fc7
```

Raw logits llama.cpp reference:

```sh
c++ -std=c++17 -O2 -I/usr/local/include scripts/reference_logits_llamacpp.cpp -L/usr/local/lib -Wl,-rpath,/usr/local/lib -lllama -lggml -lggml-cpu -lggml-base -o /tmp/reference_logits_llamacpp
cargo run --release -p det-cli -- logits -m /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf --tokens 151643,9707,151645 --dump /tmp/detllm-qwen-151643-9707-151645.rawlogits.bin --hash --threads 8
/tmp/reference_logits_llamacpp --model /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf --tokens 151643,9707,151645 --out /tmp/llamacpp-qwen-151643-9707-151645.rawlogits.bin --threads 8 --ctx-size 16 --batch-size 16 --expected-vocab 151936 --expected-rows 3 --quiet
cargo run -p xtask -- compare-logits --actual /tmp/detllm-qwen-151643-9707-151645.rawlogits.bin --reference /tmp/llamacpp-qwen-151643-9707-151645.rawlogits.bin --row-size 151936 --rows 3 --min-cosine 0.999
```

Observed output:

```text
54a2d926c73430df3b849f91ad106f0723bc5f007d851c2767e8b115e1ca7fc7
reference_logits_llamacpp rows=3 vocab=151936 values=455808
compare-logits values=455808 cosine=0.999795659 max_abs_diff=0.504380226 rms_diff=0.078664062 rows=3 row_size=151936 min_row_cosine=0.999737289
```

Minimal codec smoke:

```sh
cargo run --release -p xtask -- bench-file --model /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf --input /tmp/detllm-external/hello.txt --n-ctx 16 --iters 1
```

Observed output:

```text
bench-file model=/tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf input=/tmp/detllm-external/hello.txt limit_bytes=all limit_tokens=all iters=1 warmup=true threads=default n_ctx=16 overlap=4 model_sha256=d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8 input_sha256=66a045b452102c59d840ec097d59d9467e13a3f34f6494e539ffd32c1bb35f18
bench-file: source_input_bytes=6 measured_input_bytes=6 total_input_bytes=6 tokens=2 total_tokens=2 payload_bytes=10 dtlz_bytes=66 payload_bits_per_byte=13.333333 dtlz_bits_per_byte=88.000000 compression_ratio=11.000000 elapsed_ms=412.009 input_bytes_per_s=14.563 tokens_per_s=4.854
```

This is real Qwen2.5 GGUF evidence for GPT-2-style ByteBPE tokenizer
construction, `qwen2` metadata parsing, split-half RoPE configuration, full
required tensor compatibility on Q8_0, optional attention projection bias
loading, single-token forward, chunk-size-invariant logits hashing on a
three-token stream, a llama.cpp raw-logits cosine check, and an end-to-end codec
round-trip on a tiny byte input, and a llama.cpp perplexity-path
log-probability check. It is not a substitute for the enwik8 first-1MB
compression measurement.

### SmolLM2 External Smoke

Source:

- Repository: <https://huggingface.co/unsloth/SmolLM2-1.7B-Instruct-GGUF>
- Probed model file:
  `SmolLM2-1.7B-Instruct-Q8_0.gguf`

Observed Q8_0 intake result:

```text
model-info path=/tmp/detllm-external/SmolLM2-1.7B-Instruct-Q8_0.gguf bytes=1820414624 sha256=0f3fb091804c48a561b42a4ca1be9ce2c353017187f74c48f52299cae790abe5 metadata_prefix=false gguf_version=3 metadata=33 tensors=218 data_offset=1782432
model-info metadata key=general.architecture string=llama
model-info metadata key=general.name string=SmolLM2 1.7B Instruct
model-info metadata key=tokenizer.ggml.model string=gpt2
model-info metadata key=llama.vocab_size u32=49152
model-info metadata key=tokenizer.ggml.add_bos_token bool=false
model-info metadata key=tokenizer.ggml.bos_token_id u32=1
model-info metadata key=tokenizer.ggml.eos_token_id u32=2
model-info metadata key=tokenizer.ggml.tokens array<string>[49152]
model-info metadata key=tokenizer.ggml.merges array<string>[48900]
model-info metadata key=tokenizer.ggml.token_type array<i32>[49152]
model-info tokenizer status=ok kind=byte_bpe
model-info byte-coverage tokens=49152 single_byte=235 emittable_single_byte=235 missing=21 missing_emittable=21 missing_first=04,06,13,14,16,1d,c0,c1,f1,f2,f5,f6,f7,f8,f9,fa,... missing_emittable_first=04,06,13,14,16,1d,c0,c1,f1,f2,f5,f6,f7,f8,f9,fa,...
model-info config status=ok block_count=24 embedding_length=2048 feed_forward_length=8192 head_count=32 head_count_kv=32 context_length=8192 rope_dimension_count=64 rope_pairing=Adjacent rope_freq_base=130000.0 rms_epsilon=1e-5 attention_scale=0.125
model-info tensor-inventory total=218 encoded_bytes=1818632192 encoded_len_errors=0 F32=49 Q8_0=169
model-info vocab status=ok tokenizer=49152 model=49152 codec_max_symbols=262144
model-info required-tensors status=ok checked=218 missing=0 shape_mismatch=0 unsupported_type=0 tied_output=true
```

Metadata-prefix screening of other public SmolLM2 GGUF candidates found the
same tokenizer byte coverage gap before downloading full model payloads:

```sh
curl -L --fail --retry 3 --range 0-4194303 -o /tmp/smollm2-bartowski-prefix.gguf https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q8_0.gguf
cargo run -p xtask -- model-info --model /tmp/smollm2-bartowski-prefix.gguf --metadata-prefix
curl -L --fail --retry 3 --range 0-4194303 -o /tmp/smollm2-hftb-q4-prefix.gguf https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF/resolve/main/smollm2-1.7b-instruct-q4_k_m.gguf
cargo run -p xtask -- model-info --model /tmp/smollm2-hftb-q4-prefix.gguf --metadata-prefix
```

Observed bartowski Q8_0 prefix result:

```text
model-info path=/tmp/smollm2-bartowski-prefix.gguf bytes=4194304 sha256=a82b6f909a52c435a6a19ebd907eb8919dd5160008ed1d24e456331de1102b2b metadata_prefix=true gguf_version=3 metadata=38 tensors=218 data_offset=1782752
model-info metadata key=general.name string=Smollm2 1.7B 8k Mix7 Ep2 v2
model-info tokenizer status=ok kind=byte_bpe
model-info byte-coverage tokens=49152 single_byte=235 emittable_single_byte=235 missing=21 missing_emittable=21 missing_first=04,06,13,14,16,1d,c0,c1,f1,f2,f5,f6,f7,f8,f9,fa,... missing_emittable_first=04,06,13,14,16,1d,c0,c1,f1,f2,f5,f6,f7,f8,f9,fa,...
model-info tensor-inventory total=218 encoded_bytes=1818632192 encoded_len_errors=0 F32=49 Q8_0=169
model-info required-tensors status=ok checked=218 missing=0 shape_mismatch=0 unsupported_type=0 tied_output=true
```

Observed HuggingFaceTB Q4_K_M prefix result:

```text
model-info path=/tmp/smollm2-hftb-q4-prefix.gguf bytes=4194304 sha256=278ab31551e6bef87bdbdfdb6d283c7515e5059016f19dee4cc4c26d2d4ed8ae metadata_prefix=true gguf_version=3 metadata=34 tensors=218 data_offset=1782464
model-info metadata key=general.architecture string=llama
model-info metadata key=general.name string=Smollm2 1.7B 8k Mix7 Ep2 v2
model-info metadata key=tokenizer.ggml.model string=gpt2
model-info metadata key=llama.vocab_size u32=49152
model-info metadata key=tokenizer.ggml.add_bos_token bool=false
model-info metadata key=tokenizer.ggml.bos_token_id u32=1
model-info metadata key=tokenizer.ggml.eos_token_id u32=2
model-info metadata key=tokenizer.ggml.tokens array<string>[49152]
model-info metadata key=tokenizer.ggml.merges array<string>[48900]
model-info metadata key=tokenizer.ggml.token_type array<i32>[49152]
model-info tokenizer status=ok kind=byte_bpe
model-info byte-coverage tokens=49152 single_byte=235 emittable_single_byte=235 missing=21 missing_emittable=21 missing_first=04,06,13,14,16,1d,c0,c1,f1,f2,f5,f6,f7,f8,f9,fa,... missing_emittable_first=04,06,13,14,16,1d,c0,c1,f1,f2,f5,f6,f7,f8,f9,fa,...
model-info config status=ok block_count=24 embedding_length=2048 feed_forward_length=8192 head_count=32 head_count_kv=32 context_length=8192 rope_dimension_count=64 rope_pairing=Adjacent rope_freq_base=130000.0 rms_epsilon=1e-5 attention_scale=0.125
model-info tensor-inventory total=218 encoded_bytes=1053827072 encoded_len_errors=0 F32=49 Q4_K=144 Q6_K=25
model-info vocab status=ok tokenizer=49152 model=49152 codec_max_symbols=262144
model-info required-tensors status=ok checked=218 missing=0 shape_mismatch=0 unsupported_type=0 tied_output=true
```

Minimal logits smoke:

```sh
cargo run --release -p det-cli -- logits -m /tmp/detllm-external/SmolLM2-1.7B-Instruct-Q8_0.gguf --tokens 1 --hash --threads 8
cargo run --release -p det-cli -- logits -m /tmp/detllm-external/SmolLM2-1.7B-Instruct-Q8_0.gguf --tokens 1,2,3 --hash --chunk-size 1 --threads 8
cargo run --release -p det-cli -- logits -m /tmp/detllm-external/SmolLM2-1.7B-Instruct-Q8_0.gguf --tokens 1,2,3 --hash --chunk-size 3 --threads 8
```

Observed output:

```text
tokens=1 hash = e8baace71623c43dcf3fb2ee5be04317effd51c87f0072b2650f7e1693f86307
tokens=1,2,3 chunk-size=1 hash = 691f2b299569cf86d2a8f7a21b9bec1942ff876db0bbcb37087baab6720b25b1
tokens=1,2,3 chunk-size=3 hash = 691f2b299569cf86d2a8f7a21b9bec1942ff876db0bbcb37087baab6720b25b1
```

Raw logits llama.cpp reference:

```sh
c++ -std=c++17 -O2 -I/usr/local/include scripts/reference_logits_llamacpp.cpp -L/usr/local/lib -Wl,-rpath,/usr/local/lib -lllama -lggml -lggml-cpu -lggml-base -o /tmp/reference_logits_llamacpp
cargo run --release -p det-cli -- logits -m /tmp/detllm-external/SmolLM2-1.7B-Instruct-Q8_0.gguf --tokens 1,2,3 --dump /tmp/detllm-smollm2-123.rawlogits.bin --hash --threads 8
/tmp/reference_logits_llamacpp --model /tmp/detllm-external/SmolLM2-1.7B-Instruct-Q8_0.gguf --tokens 1,2,3 --out /tmp/llamacpp-smollm2-123.rawlogits.bin --threads 8 --ctx-size 16 --batch-size 16 --expected-vocab 49152 --expected-rows 3 --quiet
cargo run -p xtask -- compare-logits --actual /tmp/detllm-smollm2-123.rawlogits.bin --reference /tmp/llamacpp-smollm2-123.rawlogits.bin --row-size 49152 --rows 3 --min-cosine 0.999
```

Observed output:

```text
691f2b299569cf86d2a8f7a21b9bec1942ff876db0bbcb37087baab6720b25b1
reference_logits_llamacpp rows=3 vocab=49152 values=147456
compare-logits values=147456 cosine=0.999790574 max_abs_diff=0.446167946 rms_diff=0.097952058 rows=3 row_size=49152 min_row_cosine=0.999759078
```

Longer 8-token raw logits llama.cpp reference, using the current tokenizer
output for `Hello world from detllm validation.`:

```sh
cargo run --release -p det-cli -- tokenize -m /tmp/detllm-external/SmolLM2-1.7B-Instruct-Q8_0.gguf -p "Hello world from detllm validation."
cargo run --release -p det-cli -- logits -m /tmp/detllm-external/SmolLM2-1.7B-Instruct-Q8_0.gguf --tokens 19556,905,429,964,764,93,13132,30 --dump /tmp/detllm-smollm2-current-tokenized-8.rawlogits.bin --hash --threads 8
/tmp/reference_logits_llamacpp --model /tmp/detllm-external/SmolLM2-1.7B-Instruct-Q8_0.gguf --tokens 19556,905,429,964,764,93,13132,30 --out /tmp/llamacpp-smollm2-current-tokenized-8.rawlogits.bin --threads 8 --ctx-size 16 --batch-size 16 --expected-vocab 49152 --expected-rows 8 --sequential --quiet
cargo run --release -p xtask -- compare-logits --actual /tmp/detllm-smollm2-current-tokenized-8.rawlogits.bin --reference /tmp/llamacpp-smollm2-current-tokenized-8.rawlogits.bin --row-size 49152 --rows 8 --min-cosine 0.999 --worst-rows 8 --top-diffs 5
```

Observed output:

```text
19556,905,429,964,764,93,13132,30
f9b3942c20f3a4177f8d41544a918af6cc6ec90a51c085f1f69cc73cf9f6683a
reference_logits_llamacpp rows=8 vocab=49152 values=393216
compare-logits values=393216 cosine=0.999467131 max_abs_diff=0.856705666 rms_diff=0.107144161 rows=8 row_size=49152 min_row=7 min_row_cosine=0.999227139
compare-logits-worst-row row=7 cosine=0.999227139 max_abs_diff=0.441878259 rms_diff=0.116565931
compare-logits-worst-row row=3 cosine=0.999329501 max_abs_diff=0.490562439 rms_diff=0.114232291
compare-logits-worst-row row=2 cosine=0.999332700 max_abs_diff=0.856705666 rms_diff=0.168937839
compare-logits-worst-row row=5 cosine=0.999390521 max_abs_diff=0.472980499 rms_diff=0.088065795
compare-logits-worst-row row=4 cosine=0.999393480 max_abs_diff=0.464054108 rms_diff=0.084072347
compare-logits-worst-row row=6 cosine=0.999442562 max_abs_diff=0.493749619 rms_diff=0.088909721
compare-logits-worst-row row=1 cosine=0.999634599 max_abs_diff=0.547472954 rms_diff=0.084311656
compare-logits-worst-row row=0 cosine=0.999860060 max_abs_diff=0.349436283 rms_diff=0.082614803
compare-logits-top-diff rank=1 index=110078 row=2 col=11774 actual=-11.189858437 reference=-12.046564102 abs_diff=0.856705666
compare-logits-top-diff rank=2 index=138149 row=2 col=39845 actual=-4.718740940 reference=-5.537504196 abs_diff=0.818763256
compare-logits-top-diff rank=3 index=116485 row=2 col=18181 actual=-5.734763622 reference=-6.507501602 abs_diff=0.772737980
compare-logits-top-diff rank=4 index=132505 row=2 col=34201 actual=-9.688414574 reference=-10.455325127 abs_diff=0.766910553
compare-logits-top-diff rank=5 index=117723 row=2 col=19419 actual=-16.741926193 reference=-17.489261627 abs_diff=0.747335434
```

Tokenizer-backed text paths can use bytes that are present in the partial BPE
seed table:

```sh
cargo run --release -p det-cli -- tokenize -m /tmp/detllm-external/SmolLM2-1.7B-Instruct-Q8_0.gguf -p "Hello"
```

Observed output:

```text
19556
```

Codec paths now use deterministic byte escapes for bytes missing from the BPE
seed set. Minimal arbitrary-byte codec smoke with bytes that SmolLM2 cannot
emit as single-byte vocabulary tokens:

```sh
printf 'detllm\xff\xc0\x04\n' > /tmp/detllm-smollm2-escape.bin
cargo run --release -p xtask -- bench-file --model /tmp/detllm-external/SmolLM2-1.7B-Instruct-Q8_0.gguf --input /tmp/detllm-smollm2-escape.bin --n-ctx 16 --iters 1 --no-warmup --show-phases
```

Observed output:

```text
bench-file model=/tmp/detllm-external/SmolLM2-1.7B-Instruct-Q8_0.gguf input=/tmp/detllm-smollm2-escape.bin limit_bytes=all limit_tokens=all iters=1 warmup=false threads=default n_ctx=16 overlap=4 model_sha256=0f3fb091804c48a561b42a4ca1be9ce2c353017187f74c48f52299cae790abe5 input_sha256=85932b70980ded4c5fc6a3b73b47839b3e4fcb65a51a7f164b5b267e8da02a71
bench-file: source_input_bytes=10 measured_input_bytes=10 total_input_bytes=10 tokenized_tokens=7 tokens=7 total_tokens=7 payload_bytes=22 dtlz_bytes=78 payload_bits_per_byte=17.600000 dtlz_bits_per_byte=62.400000 compression_ratio=7.800000 elapsed_ms=1581.394 input_bytes_per_s=6.324 tokens_per_s=4.426
bench-file-phases: model_read_ms=3274.449 gguf_parse_ms=6.978 model_load_ms=2615.510 tokenizer_setup_ms=104.751 input_read_ms=0.050 tokenize_ms=1.346 token_prefix_ms=0.000 warmup_ms=0.000 measured_ms=1581.394 total_ms=13855.380
```

The public CLI writes the byte-escape flag in new DTLZ files and restores the
same bytes:

```sh
cargo run --release -p det-cli -- compress -m /tmp/detllm-external/SmolLM2-1.7B-Instruct-Q8_0.gguf -i /tmp/detllm-smollm2-escape.bin -o /tmp/detllm-smollm2-escape.dtlz --n-ctx 16 --threads 8
cargo run --release -p det-cli -- decompress -m /tmp/detllm-external/SmolLM2-1.7B-Instruct-Q8_0.gguf -i /tmp/detllm-smollm2-escape.dtlz -o /tmp/detllm-smollm2-escape.restored --threads 8
xxd -p -l 8 /tmp/detllm-smollm2-escape.dtlz
cmp /tmp/detllm-smollm2-escape.bin /tmp/detllm-smollm2-escape.restored
sha256sum /tmp/detllm-smollm2-escape.bin /tmp/detllm-smollm2-escape.restored /tmp/detllm-smollm2-escape.dtlz
wc -c /tmp/detllm-smollm2-escape.bin /tmp/detllm-smollm2-escape.restored /tmp/detllm-smollm2-escape.dtlz
```

Observed output:

```text
44544c5a01000100
85932b70980ded4c5fc6a3b73b47839b3e4fcb65a51a7f164b5b267e8da02a71  /tmp/detllm-smollm2-escape.bin
85932b70980ded4c5fc6a3b73b47839b3e4fcb65a51a7f164b5b267e8da02a71  /tmp/detllm-smollm2-escape.restored
f238623da90e2452fcce0370fe4dfe516aa715f84a08301f128cb6d3d5837116  /tmp/detllm-smollm2-escape.dtlz
10 /tmp/detllm-smollm2-escape.bin
10 /tmp/detllm-smollm2-escape.restored
78 /tmp/detllm-smollm2-escape.dtlz
98 total
```

This is real SmolLM2 GGUF evidence for model config parsing, required tensor
compatibility on Q8_0, single-token forward, chunk-size-invariant logits
hashing on a three-token stream, a three-token llama.cpp raw-logits cosine
check that passes the 0.999 target, and an 8-token tokenizer-backed
raw-logits check that passes the 0.999 per-row target. It also records that
partial GPT-2 BPE
tokenizer construction is usable for present-byte text, while the tested full
GGUF and the two metadata-prefix-screened public candidates expose only 235 of
the 256 byte values as single-byte BPE seed tokens. The byte escape tail keeps
arbitrary-byte codec round-trip possible without changing the model vocabulary.

### Target-Model Round-Trip Matrix

The external target-model round-trip matrix runs the public `compress` /
`decompress` CLI over empty, multilingual UTF-8, binary-mixed, and
context-spanning inputs for every currently tracked target GGUF. The
`context-spanning` input tokenizes to 13 tokens for each target tokenizer, so
`--n-ctx 8` forces at least one window rollover. The script runs `cmp` for every
restored file before printing hashes and byte counts.

Command:

```sh
cargo build --release -p det-cli --features parallel,simd
scripts/run-target-roundtrip-matrix.sh \
  --tinyllama-q8 /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  --tinyllama-q4 /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  --qwen25-q8 /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf \
  --smollm2-q8 /tmp/detllm-external/SmolLM2-1.7B-Instruct-Q8_0.gguf \
  --out /tmp/detllm-roundtrip-matrix-small \
  --threads 8 \
  --n-ctx 8
```

Observed inputs:

| input | bytes | SHA-256 |
|---|---:|---|
| `empty` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `multilingual` | 104 | `6ebdf0dfe422b6a7c8db30204a579809e147b99b55d22d91b105388bd5535f9e` |
| `binary-mixed` | 28 | `458b71a7d9440b62ec2e34688a788980d90a4d872151d0634bb8e5402108b5a8` |
| `context-spanning` | 64 | `3f29407e1529b5ba5f001e09a1d1e53b371a99036bfcc395c041d4cc23e75147` |

Observed output:

| model | input | DTLZ bytes | DTLZ SHA-256 | restored bytes | restored SHA-256 |
|---|---|---:|---|---:|---|
| TinyLlama Q8_0 | empty | 64 | `f5cddfa0c666838a9a4931953caf92ae82b895d836d054bb3d10a436846d03ab` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| TinyLlama Q8_0 | multilingual | 125 | `8ca05a0bbe0682aa64fa10dbf6133cf275f89b5ec1d35a10ea39e9a2d9adec8e` | 104 | `6ebdf0dfe422b6a7c8db30204a579809e147b99b55d22d91b105388bd5535f9e` |
| TinyLlama Q8_0 | binary-mixed | 86 | `0c8551a3afa977fe51e802bc5a4810925b2707e720ed74e5cf9057f07c421092` | 28 | `458b71a7d9440b62ec2e34688a788980d90a4d872151d0634bb8e5402108b5a8` |
| TinyLlama Q8_0 | context-spanning | 73 | `d0a0b9cb671df18d6188c5bb53487a085e65869ceee07f94fe5a768a123337ee` | 64 | `3f29407e1529b5ba5f001e09a1d1e53b371a99036bfcc395c041d4cc23e75147` |
| TinyLlama Q4_0 | empty | 64 | `a769ef53bc5f0f9cf20875c9f916e3b77bd7927166c887f737208dcfbebfc1ad` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| TinyLlama Q4_0 | multilingual | 125 | `8c74fc6391996009709f78f2f5188af594ad596d4b22f51093dfd65b49986de8` | 104 | `6ebdf0dfe422b6a7c8db30204a579809e147b99b55d22d91b105388bd5535f9e` |
| TinyLlama Q4_0 | binary-mixed | 85 | `d2f89b70a1681bd5aaf28309e1bbc3d1f109c8ebba2c432875b7ef1b19229516` | 28 | `458b71a7d9440b62ec2e34688a788980d90a4d872151d0634bb8e5402108b5a8` |
| TinyLlama Q4_0 | context-spanning | 73 | `e79297e6e0da6e4449833057d0aaf6a6bb2b6cefe8764bc02bccb39f613f8395` | 64 | `3f29407e1529b5ba5f001e09a1d1e53b371a99036bfcc395c041d4cc23e75147` |
| Qwen2.5 Q8_0 | empty | 64 | `5edd561a793bf230a3de8a165ef37942f8f85469488b5cc848a5fe626a0ea5e2` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| Qwen2.5 Q8_0 | multilingual | 91 | `ba121249e1b5118fe5934c40fd50633c66350816fe3da34739e535d31a6009f7` | 104 | `6ebdf0dfe422b6a7c8db30204a579809e147b99b55d22d91b105388bd5535f9e` |
| Qwen2.5 Q8_0 | binary-mixed | 85 | `ea719f3444398e1e1352aee5a4ac6690ae40ce106dc1990a4a3c60a3cbe7a72c` | 28 | `458b71a7d9440b62ec2e34688a788980d90a4d872151d0634bb8e5402108b5a8` |
| Qwen2.5 Q8_0 | context-spanning | 69 | `7047a35e2c976cb35333e2ccc653552f94a58e77d1a884719a703d6f8b2b1fa5` | 64 | `3f29407e1529b5ba5f001e09a1d1e53b371a99036bfcc395c041d4cc23e75147` |
| SmolLM2 Q8_0 | empty | 64 | `d1370c44fc46be82cca9d7075d22ecae99e582694d89d0cc192595317786e1af` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| SmolLM2 Q8_0 | multilingual | 95 | `68e4bedb09e4f1cefc1cbff7659dab60579229220740e0e350d8b5949ad5476e` | 104 | `6ebdf0dfe422b6a7c8db30204a579809e147b99b55d22d91b105388bd5535f9e` |
| SmolLM2 Q8_0 | binary-mixed | 87 | `2ac0d09372c2f16a57209a5e5bc585c8fb47913ff2bcb77588561101a61be4a4` | 28 | `458b71a7d9440b62ec2e34688a788980d90a4d872151d0634bb8e5402108b5a8` |
| SmolLM2 Q8_0 | context-spanning | 70 | `960025e02a6baa87218018b877266628e37800585e5839a72d7fde6671f0d1c0` | 64 | `3f29407e1529b5ba5f001e09a1d1e53b371a99036bfcc395c041d4cc23e75147` |

This covers the current §9.7 target-model matrix over empty, multilingual,
binary-mixed, and context-spanning inputs for the tracked model/quantization
set. Larger arbitrary-byte and multi-window payloads can still be used as
stress tests, but this matrix is the acceptance smoke for each target model.

### Target-Model Codec Determinism Matrix

The target-model codec determinism matrix checks that public `compress` output
is bit-identical across thread-count settings and binary builds. It uses the
byte-escape `binary-mixed` input and the `context-spanning` input from the
round-trip matrix, with `--n-ctx 8`, both the default scalar build and a
`parallel,simd` build, and `threads=1,2,7,16`. Every DTLZ file is decompressed
and compared with the original input before the matrix accepts the row. This
matches the thread-count and backend build set from `detllm-design.md` §9.2.

Command:

```sh
cargo build --release -p det-cli
cp target/release/detllm /tmp/detllm-scalar
cargo build --release -p det-cli --features parallel,simd
cp target/release/detllm /tmp/detllm-parallel-simd
scripts/run-target-codec-determinism-matrix.sh \
  --bin /tmp/detllm-scalar \
  --extra-bin parallel-simd=/tmp/detllm-parallel-simd \
  --tinyllama-q8 /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  --tinyllama-q4 /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  --qwen25-q8 /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf \
  --smollm2-q8 /tmp/detllm-external/SmolLM2-1.7B-Instruct-Q8_0.gguf
```

Observed inputs:

| input | bytes | SHA-256 |
|---|---:|---|
| `binary-mixed` | 28 | `458b71a7d9440b62ec2e34688a788980d90a4d872151d0634bb8e5402108b5a8` |
| `context-spanning` | 64 | `3f29407e1529b5ba5f001e09a1d1e53b371a99036bfcc395c041d4cc23e75147` |

Observed output summary:

| model | input | invariant settings | DTLZ bytes | DTLZ SHA-256 |
|---|---|---:|---:|---|
| TinyLlama Q8_0 | binary-mixed | 8 | 86 | `0c8551a3afa977fe51e802bc5a4810925b2707e720ed74e5cf9057f07c421092` |
| TinyLlama Q8_0 | context-spanning | 8 | 73 | `d0a0b9cb671df18d6188c5bb53487a085e65869ceee07f94fe5a768a123337ee` |
| TinyLlama Q4_0 | binary-mixed | 8 | 85 | `d2f89b70a1681bd5aaf28309e1bbc3d1f109c8ebba2c432875b7ef1b19229516` |
| TinyLlama Q4_0 | context-spanning | 8 | 73 | `e79297e6e0da6e4449833057d0aaf6a6bb2b6cefe8764bc02bccb39f613f8395` |
| Qwen2.5 Q8_0 | binary-mixed | 8 | 85 | `ea719f3444398e1e1352aee5a4ac6690ae40ce106dc1990a4a3c60a3cbe7a72c` |
| Qwen2.5 Q8_0 | context-spanning | 8 | 69 | `7047a35e2c976cb35333e2ccc653552f94a58e77d1a884719a703d6f8b2b1fa5` |
| SmolLM2 Q8_0 | binary-mixed | 8 | 87 | `2ac0d09372c2f16a57209a5e5bc585c8fb47913ff2bcb77588561101a61be4a4` |
| SmolLM2 Q8_0 | context-spanning | 8 | 70 | `960025e02a6baa87218018b877266628e37800585e5839a72d7fde6671f0d1c0` |

This reuses the DTLZ hashes from the target round-trip matrix and proves that
the current target-model codec path emits the same payload across
`2 binary builds * 4 thread counts` for byte-escape and context-rollover
cases. This broadens the target-model codec evidence for M5, M6, and M8, but
it is still not the final M4 full first-1MB target-model compression-rate run.

### Target-Model Determinism Matrix

The target-model determinism matrix checks that the current external GGUF set
keeps identical logits bytes across deterministic chunking and thread-count
settings. It uses the same tokenizer-backed 8-token streams as the raw-logits
reference matrix and compares the public `detllm logits --hash` output across
both a default scalar build and a `parallel,simd` build, with
`threads=1,2,7,16` and `chunk-size=1,3,8`. The `chunk-size=8` setting is the
full-stream prefill case for these 8-token streams. `--extra-bin LABEL=PATH`
adds another binary to the same comparison, so the script can cover the
`detllm-design.md` §9.2 backend set without relying on separate manual runs.

Command:

```sh
cargo build --release -p det-cli
cp target/release/detllm /tmp/detllm-scalar
cargo build --release -p det-cli --features parallel,simd
cp target/release/detllm /tmp/detllm-parallel-simd
scripts/run-target-determinism-matrix.sh \
  --bin /tmp/detllm-scalar \
  --extra-bin parallel-simd=/tmp/detllm-parallel-simd \
  --tinyllama-q8 /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  --tinyllama-q4 /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  --qwen25-q8 /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf \
  --smollm2-q8 /tmp/detllm-external/SmolLM2-1.7B-Instruct-Q8_0.gguf
```

Observed output summary:

| model | invariant settings | logits hash |
|---|---:|---|
| TinyLlama Q8_0 | 24 | `ded3a5204a66f58e529101511fe8d2e051fe9d71897d930ea49ec57372f3001a` |
| TinyLlama Q4_0 | 24 | `da312ede8d5c3ac7599987204c7ba954f3d86315c259c7f6c3838040cf95efb5` |
| Qwen2.5 Q8_0 | 24 | `22a98865d5bd6c45a2ae4c1a29e8b37db58a78a6c7c8caedb53a3d6baee33088` |
| SmolLM2 Q8_0 | 24 | `f9b3942c20f3a4177f8d41544a918af6cc6ec90a51c085f1f69cc73cf9f6683a` |

Each row passed all `2 binary builds * 4 thread counts * 3 chunk sizes`
combinations bit-for-bit. This broadens the target-model evidence for DET-2,
M5, M6, and M8, but it is not a cross-platform target-model run; CI
cross-platform hash checking still uses the bundled fixtures.

## File Codec Bench Harness

Command:

```sh
cargo run --release -p xtask -- bench-file --model testdata/tiny-f32.gguf --input testdata/tiny.tokens.txt --n-ctx 8 --iters 1
cargo run --release -p xtask --features parallel,simd -- bench-file --model model.gguf --input enwik8 --limit-bytes 4096 --limit-tokens 512 --n-ctx 2048 --threads 8 --iters 1 --no-warmup
cargo run --release -p xtask --features parallel,simd -- bench-file --model model.gguf --input enwik8 --limit-bytes 1048576 --n-ctx 2048 --threads 8 --iters 1 --no-warmup --encode-only --show-phases --summary bench-file.summary --progress-every 100 --progress-summary bench-file.progress
```

Build `xtask` with `--features parallel,simd` for target-model benchmark
commands. The `parallel` feature forwards to `det-model/parallel`, so
`--threads N` engages deterministic row-parallel GEMV; `simd` forwards to the
quantized kernel feature.
For tokenizers with incomplete byte coverage, `bench-file` counts byte escapes
as codec symbols in the `tokenized_tokens`, `tokens`, and `total_tokens`
fields; for complete byte-coverage tokenizers these remain ordinary tokenizer
token counts.
Long target-model compression-rate preflights can use `--encode-only` after a
separate round-trip smoke has established codec correctness. This mode measures
payload generation and compression ratio without paying for the mirrored decode
pass. The default remains `mode=round-trip` and verifies decoded bytes.
`--progress-summary PATH` atomically writes the latest `bench-file-progress`
line to a file while the run is still active. This is separate from
`--summary PATH`, which is written only after the measured loop finishes.
When a token-prefix preflight uses `--limit-tokens`, `--estimate-full-run`
adds a `bench-file-estimate` line that scales measured token throughput to the
full tokenized input prefix. This is an ETA and planning aid, not acceptance
evidence for the full compression-rate result.

Observed smoke output on the bundled token text fixture:

| model | model SHA-256 | input SHA-256 | measured input bytes | tokens | payload bytes | DTLZ bytes | payload bpb | DTLZ bpb | ratio |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `testdata/tiny-f32.gguf` | `ce2aa01900a63585a409ef995a2827dcac81e1678e38a1ab0733302ba82ce79b` | `bfdf7888835d22d01ce148aa49e1e766f11e3fbe8631f08215e1c9173270dbd8` | 12 | 12 | 19 | 75 | 12.666667 | 50.000000 | 6.250000 |
| `testdata/tiny-qmix.gguf` | `4adbef1f9806fb17050d4520135bf8c8b4308840637b2e27589887f7fc03338f` | `bfdf7888835d22d01ce148aa49e1e766f11e3fbe8631f08215e1c9173270dbd8` | 12 | 12 | 19 | 75 | 12.666667 | 50.000000 | 6.250000 |

Observed enwik8 first-1MB fixture measurement:

```sh
curl -L --fail --retry 3 -o /tmp/enwik8.zip http://mattmahoney.net/dc/enwik8.zip
unzip -o /tmp/enwik8.zip -d /tmp
sha256sum /tmp/enwik8.zip /tmp/enwik8
cargo run --release -p xtask -- bench-file --model testdata/tiny-f32.gguf --input /tmp/enwik8 --limit-bytes 1048576 --n-ctx 8 --iters 1
```

```text
547994d9980ebed1288380d652999f38a14fe291a6247c157c3d33d4932534bc  /tmp/enwik8.zip
2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8  /tmp/enwik8
bench-file model=testdata/tiny-f32.gguf input=/tmp/enwik8 limit_bytes=1048576 limit_tokens=all iters=1 warmup=true threads=default n_ctx=8 overlap=2 model_sha256=ce2aa01900a63585a409ef995a2827dcac81e1678e38a1ab0733302ba82ce79b input_sha256=4fb5efa9f35df431737731bf3c8f38a467b69731940ff82a4ee0e218aae58834
bench-file: source_input_bytes=100000000 measured_input_bytes=1048576 total_input_bytes=1048576 tokens=1048576 total_tokens=1048576 payload_bytes=1048535 dtlz_bytes=1048591 payload_bits_per_byte=7.999687 dtlz_bits_per_byte=8.000114 compression_ratio=1.000014 elapsed_ms=51811.324 input_bytes_per_s=20238.356 tokens_per_s=20238.356
```

Observed token-prefix smoke on the bundled token text fixture:

```sh
cargo run -p xtask -- bench-file --model testdata/tiny-f32.gguf --input testdata/tiny.tokens.txt --n-ctx 8 --iters 1 --limit-tokens 5 --no-warmup
```

```text
bench-file model=testdata/tiny-f32.gguf input=testdata/tiny.tokens.txt limit_bytes=all limit_tokens=5 iters=1 warmup=false threads=default n_ctx=8 overlap=2 model_sha256=ce2aa01900a63585a409ef995a2827dcac81e1678e38a1ab0733302ba82ce79b input_sha256=c0be322c1ad6af50f418b96232d98fe25a36d5d0a557291833f8248f2084b8ef
bench-file: source_input_bytes=12 measured_input_bytes=5 total_input_bytes=5 tokens=5 total_tokens=5 payload_bytes=12 dtlz_bytes=68 payload_bits_per_byte=19.200000 dtlz_bits_per_byte=108.800000 compression_ratio=13.600000 elapsed_ms=0.872 input_bytes_per_s=5736.261 tokens_per_s=5736.261
```

Observed TinyLlama Q8_0 target-model token-prefix smoke on the canonical enwik8
stream:

```sh
cargo run --release -p xtask --features parallel,simd -- bench-file --model /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q8_0.gguf --input /tmp/enwik8 --limit-bytes 1048576 --limit-tokens 16 --n-ctx 64 --threads 8 --iters 1 --no-warmup
```

```text
bench-file model=/tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q8_0.gguf input=/tmp/enwik8 limit_bytes=1048576 limit_tokens=16 iters=1 warmup=false threads=8 n_ctx=64 overlap=16 model_sha256=a4c9bb1dbaa372f6381a035fa5c02ef087aaa1ff1f843a56a22328114f03fc59 input_sha256=516d4b5d16ec0a573a2eaf415abe04b40ca38038fce3531e71bb3f019ff9b6de
bench-file: source_input_bytes=100000000 measured_input_bytes=47 total_input_bytes=47 tokens=16 total_tokens=16 payload_bytes=33 dtlz_bytes=89 payload_bits_per_byte=5.617021 dtlz_bits_per_byte=15.148936 compression_ratio=1.893617 elapsed_ms=29911.665 input_bytes_per_s=1.571 tokens_per_s=0.535
```

Observed Qwen2.5 Q8_0 target-model token-prefix smoke on the canonical enwik8
stream:

```sh
cargo run --release -p xtask --features parallel,simd -- bench-file --model /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf --input /tmp/enwik8 --limit-bytes 1048576 --limit-tokens 16 --n-ctx 64 --threads 8 --iters 1 --no-warmup --show-phases --progress-every 8
```

```text
bench-file-progress phase=encode tokens_done=8 tokens_total=16 elapsed_ms=1024.279 tokens_per_s=7.810
bench-file-progress phase=encode tokens_done=16 tokens_total=16 elapsed_ms=2068.217 tokens_per_s=7.736
bench-file-progress phase=decode tokens_done=8 tokens_total=16 elapsed_ms=1035.963 tokens_per_s=7.722
bench-file-progress phase=decode tokens_done=16 tokens_total=16 elapsed_ms=2049.644 tokens_per_s=7.806
bench-file model=/tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf input=/tmp/enwik8 limit_bytes=1048576 limit_tokens=16 iters=1 warmup=false threads=8 n_ctx=64 overlap=16 model_sha256=d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8 input_sha256=4fe5a21798e43c8258edcf9f3a98fac2df77613b4d2add15a2a3082eedc7b0b2
bench-file: source_input_bytes=100000000 measured_input_bytes=53 total_input_bytes=53 tokens=16 total_tokens=16 payload_bytes=14 dtlz_bytes=70 payload_bits_per_byte=2.113208 dtlz_bits_per_byte=10.566038 compression_ratio=1.320755 elapsed_ms=4245.398 input_bytes_per_s=12.484 tokens_per_s=3.769
bench-file-phases: model_read_ms=2203.318 gguf_parse_ms=30.830 model_load_ms=2247.226 tokenizer_setup_ms=284.216 input_read_ms=118.886 tokenize_ms=913.220 token_prefix_ms=9.846 warmup_ms=0.000 measured_ms=4245.398 total_ms=15570.105
```

After `bench-file` learned to read only the requested byte prefix for
`--limit-bytes`, the same Qwen2.5 path was extended to a 64-token
encode/decode round-trip:

```sh
cargo run --release -p xtask --features parallel,simd -- bench-file --model /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf --input /tmp/enwik8 --limit-bytes 1048576 --limit-tokens 64 --n-ctx 128 --threads 8 --iters 1 --no-warmup --show-phases --progress-every 16
```

```text
bench-file-progress phase=encode tokens_done=16 tokens_total=64 elapsed_ms=2826.319 tokens_per_s=5.661
bench-file-progress phase=encode tokens_done=32 tokens_total=64 elapsed_ms=5793.098 tokens_per_s=5.524
bench-file-progress phase=encode tokens_done=48 tokens_total=64 elapsed_ms=8710.002 tokens_per_s=5.511
bench-file-progress phase=encode tokens_done=64 tokens_total=64 elapsed_ms=11635.372 tokens_per_s=5.500
bench-file-progress phase=decode tokens_done=16 tokens_total=64 elapsed_ms=2709.283 tokens_per_s=5.906
bench-file-progress phase=decode tokens_done=32 tokens_total=64 elapsed_ms=5549.411 tokens_per_s=5.766
bench-file-progress phase=decode tokens_done=48 tokens_total=64 elapsed_ms=8507.142 tokens_per_s=5.642
bench-file-progress phase=decode tokens_done=64 tokens_total=64 elapsed_ms=11413.044 tokens_per_s=5.608
bench-file model=/tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf input=/tmp/enwik8 limit_bytes=1048576 limit_tokens=64 iters=1 warmup=false threads=8 n_ctx=128 overlap=32 model_sha256=d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8 input_sha256=b4997b129849e53a0cb6265f2561d8e57ad57003ffbcc1c7357b03918e79b03b
bench-file: source_input_bytes=100000000 measured_input_bytes=190 total_input_bytes=190 tokens=64 total_tokens=64 payload_bytes=15 dtlz_bytes=71 payload_bits_per_byte=0.631579 dtlz_bits_per_byte=2.989474 compression_ratio=0.373684 elapsed_ms=23201.015 input_bytes_per_s=8.189 tokens_per_s=2.758
bench-file-phases: model_read_ms=2326.019 gguf_parse_ms=33.568 model_load_ms=2536.310 tokenizer_setup_ms=436.087 input_read_ms=45.359 tokenize_ms=1412.359 token_prefix_ms=0.134 warmup_ms=0.000 measured_ms=23201.015 total_ms=36310.746
```

Current-format Qwen2.5 encode-only preflight on the same 64-token prefix:

```sh
cargo run --release -p xtask --features parallel,simd -- bench-file --model /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf --input /tmp/enwik8 --limit-bytes 1048576 --limit-tokens 64 --n-ctx 128 --threads 8 --iters 1 --no-warmup --encode-only --show-phases --progress-every 16
```

```text
bench-file-progress phase=encode tokens_done=16 tokens_total=64 elapsed_ms=2758.847 tokens_per_s=5.800
bench-file-progress phase=encode tokens_done=32 tokens_total=64 elapsed_ms=5764.774 tokens_per_s=5.551
bench-file-progress phase=encode tokens_done=48 tokens_total=64 elapsed_ms=8821.398 tokens_per_s=5.441
bench-file-progress phase=encode tokens_done=64 tokens_total=64 elapsed_ms=11889.978 tokens_per_s=5.383
bench-file model=/tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf input=/tmp/enwik8 limit_bytes=1048576 limit_tokens=64 iters=1 warmup=false mode=encode-only threads=8 n_ctx=128 overlap=32 model_sha256=d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8 input_sha256=b4997b129849e53a0cb6265f2561d8e57ad57003ffbcc1c7357b03918e79b03b
bench-file: source_input_bytes=100000000 measured_input_bytes=190 total_input_bytes=190 tokenized_tokens=279472 tokens=64 total_tokens=64 payload_bytes=15 dtlz_bytes=71 payload_bits_per_byte=0.631579 dtlz_bits_per_byte=2.989474 compression_ratio=0.373684 elapsed_ms=11978.087 input_bytes_per_s=15.862 tokens_per_s=5.343
bench-file-phases: model_read_ms=2357.780 gguf_parse_ms=26.854 model_load_ms=2618.833 tokenizer_setup_ms=454.859 input_read_ms=30.442 tokenize_ms=1544.375 token_prefix_ms=0.163 warmup_ms=0.000 measured_ms=11978.087 total_ms=25361.387
```

This preserves the same measured byte prefix and payload size as the previous
round-trip run while removing the decode phase from the measured loop.

Qwen2.5 16-token encode-only preflight with full-token estimate enabled:

```sh
cargo run --release -p xtask --features parallel,simd -- bench-file --model /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf --input /tmp/enwik8 --limit-bytes 1048576 --limit-tokens 16 --n-ctx 64 --threads 8 --iters 1 --no-warmup --encode-only --show-phases --progress-every 8 --estimate-full-run
```

```text
bench-file-progress phase=encode tokens_done=8 tokens_total=16 elapsed_ms=1393.335 tokens_per_s=5.742
bench-file-progress phase=encode tokens_done=16 tokens_total=16 elapsed_ms=2842.047 tokens_per_s=5.630
bench-file model=/tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf input=/tmp/enwik8 limit_bytes=1048576 limit_tokens=16 iters=1 warmup=false mode=encode-only threads=8 n_ctx=64 overlap=16 model_sha256=d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8 input_sha256=4fe5a21798e43c8258edcf9f3a98fac2df77613b4d2add15a2a3082eedc7b0b2
bench-file: source_input_bytes=100000000 measured_input_bytes=53 total_input_bytes=53 tokenized_tokens=279472 tokens=16 total_tokens=16 payload_bytes=13 dtlz_bytes=69 payload_bits_per_byte=1.962264 dtlz_bits_per_byte=10.415094 compression_ratio=1.301887 elapsed_ms=2942.089 input_bytes_per_s=18.014 tokens_per_s=5.438
bench-file-estimate: full_tokens=279472 full_input_bytes=1048576 measured_tokens=16 scale_factor=17467.000000 estimated_measured_ms=51389459.952 estimated_measured_s=51389.460 measured_tokens_per_s=5.438
bench-file-phases: model_read_ms=1580.194 gguf_parse_ms=25.450 model_load_ms=2225.827 tokenizer_setup_ms=382.699 input_read_ms=24.083 tokenize_ms=1622.583 token_prefix_ms=0.151 warmup_ms=0.000 measured_ms=2942.089 total_ms=14987.132
```

On this host, the ETA line puts the Qwen2.5 Q8_0 first-1MB encode-only
measured loop at roughly 14 hours. It explains why the final M4 measurement is
still tracked separately instead of being folded into routine validation.

Scripted target-model prefix benchmark smoke across the current external GGUF
validation set:

```sh
scripts/run-target-bench-smoke.sh --input /tmp/enwik8 --tinyllama-q8 /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q8_0.gguf --tinyllama-q4 /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --qwen25-q8 /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf --smollm2-q8 /tmp/detllm-external/SmolLM2-1.7B-Instruct-Q8_0.gguf
```

Environment: same local `x86_64` WSL2 host as the fixture benchmark snapshot,
with `rustc 1.95.0`. Command defaults were enwik8 `--limit-bytes 1048576`,
`--limit-tokens 16`, `--encode-only`, `--threads 8`, `--n-ctx 64`,
`--iters 1`, `--no-warmup`, `--show-phases`, `--estimate-full-run`, and
`--progress-every 8`.

```text
== tinyllama-q8 ==
bench-file-progress phase=encode tokens_done=8 tokens_total=16 elapsed_ms=1036.155 tokens_per_s=7.721
bench-file-progress phase=encode tokens_done=16 tokens_total=16 elapsed_ms=2134.613 tokens_per_s=7.496
bench-file model=/tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q8_0.gguf input=/tmp/enwik8 limit_bytes=1048576 limit_tokens=16 iters=1 warmup=false mode=encode-only threads=8 n_ctx=64 overlap=16 model_sha256=a4c9bb1dbaa372f6381a035fa5c02ef087aaa1ff1f843a56a22328114f03fc59 input_sha256=516d4b5d16ec0a573a2eaf415abe04b40ca38038fce3531e71bb3f019ff9b6de
bench-file: source_input_bytes=100000000 measured_input_bytes=47 total_input_bytes=47 tokenized_tokens=336344 tokens=16 total_tokens=16 payload_bytes=33 dtlz_bytes=89 payload_bits_per_byte=5.617021 dtlz_bits_per_byte=15.148936 compression_ratio=1.893617 elapsed_ms=2185.280 input_bytes_per_s=21.508 tokens_per_s=7.322
bench-file-estimate: full_tokens=336344 full_input_bytes=1048576 measured_tokens=16 scale_factor=21021.500000 estimated_measured_ms=45937863.163 estimated_measured_s=45937.863 measured_tokens_per_s=7.322
bench-file-phases: model_read_ms=1259.832 gguf_parse_ms=8.619 model_load_ms=1405.274 tokenizer_setup_ms=31.427 input_read_ms=1.529 tokenize_ms=927.340 token_prefix_ms=0.119 warmup_ms=0.000 measured_ms=2185.280 total_ms=9697.857
== tinyllama-q4 ==
bench-file-progress phase=encode tokens_done=8 tokens_total=16 elapsed_ms=1552.729 tokens_per_s=5.152
bench-file-progress phase=encode tokens_done=16 tokens_total=16 elapsed_ms=3065.533 tokens_per_s=5.219
bench-file model=/tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q4_0.gguf input=/tmp/enwik8 limit_bytes=1048576 limit_tokens=16 iters=1 warmup=false mode=encode-only threads=8 n_ctx=64 overlap=16 model_sha256=da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556 input_sha256=516d4b5d16ec0a573a2eaf415abe04b40ca38038fce3531e71bb3f019ff9b6de
bench-file: source_input_bytes=100000000 measured_input_bytes=47 total_input_bytes=47 tokenized_tokens=336344 tokens=16 total_tokens=16 payload_bytes=34 dtlz_bytes=90 payload_bits_per_byte=5.787234 dtlz_bits_per_byte=15.319149 compression_ratio=1.914894 elapsed_ms=3096.543 input_bytes_per_s=15.178 tokens_per_s=5.167
bench-file-estimate: full_tokens=336344 full_input_bytes=1048576 measured_tokens=16 scale_factor=21021.500000 estimated_measured_ms=65093979.158 estimated_measured_s=65093.979 measured_tokens_per_s=5.167
bench-file-phases: model_read_ms=435.598 gguf_parse_ms=8.402 model_load_ms=591.899 tokenizer_setup_ms=32.735 input_read_ms=0.753 tokenize_ms=873.588 token_prefix_ms=0.126 warmup_ms=0.000 measured_ms=3096.543 total_ms=7089.378
== qwen25-q8 ==
bench-file-progress phase=encode tokens_done=8 tokens_total=16 elapsed_ms=1393.335 tokens_per_s=5.742
bench-file-progress phase=encode tokens_done=16 tokens_total=16 elapsed_ms=2842.047 tokens_per_s=5.630
bench-file model=/tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf input=/tmp/enwik8 limit_bytes=1048576 limit_tokens=16 iters=1 warmup=false mode=encode-only threads=8 n_ctx=64 overlap=16 model_sha256=d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8 input_sha256=4fe5a21798e43c8258edcf9f3a98fac2df77613b4d2add15a2a3082eedc7b0b2
bench-file: source_input_bytes=100000000 measured_input_bytes=53 total_input_bytes=53 tokenized_tokens=279472 tokens=16 total_tokens=16 payload_bytes=13 dtlz_bytes=69 payload_bits_per_byte=1.962264 dtlz_bits_per_byte=10.415094 compression_ratio=1.301887 elapsed_ms=2942.089 input_bytes_per_s=18.014 tokens_per_s=5.438
bench-file-estimate: full_tokens=279472 full_input_bytes=1048576 measured_tokens=16 scale_factor=17467.000000 estimated_measured_ms=51389459.952 estimated_measured_s=51389.460 measured_tokens_per_s=5.438
bench-file-phases: model_read_ms=1580.194 gguf_parse_ms=25.450 model_load_ms=2225.827 tokenizer_setup_ms=382.699 input_read_ms=24.083 tokenize_ms=1622.583 token_prefix_ms=0.151 warmup_ms=0.000 measured_ms=2942.089 total_ms=14987.132
== smollm2-q8 ==
bench-file-progress phase=encode tokens_done=8 tokens_total=16 elapsed_ms=1399.829 tokens_per_s=5.715
bench-file-progress phase=encode tokens_done=16 tokens_total=16 elapsed_ms=2748.361 tokens_per_s=5.822
bench-file model=/tmp/detllm-external/SmolLM2-1.7B-Instruct-Q8_0.gguf input=/tmp/enwik8 limit_bytes=1048576 limit_tokens=16 iters=1 warmup=false mode=encode-only threads=8 n_ctx=64 overlap=16 model_sha256=0f3fb091804c48a561b42a4ca1be9ce2c353017187f74c48f52299cae790abe5 input_sha256=cde97e63d212a07d92a900e45f13c3974ec428806f1d8dbf44a1ddd0083edc8d
bench-file: source_input_bytes=100000000 measured_input_bytes=46 total_input_bytes=46 tokenized_tokens=302781 tokens=16 total_tokens=16 payload_bytes=14 dtlz_bytes=70 payload_bits_per_byte=2.434783 dtlz_bits_per_byte=12.173913 compression_ratio=1.521739 elapsed_ms=2860.649 input_bytes_per_s=16.080 tokens_per_s=5.593
bench-file-estimate: full_tokens=302781 full_input_bytes=1048576 measured_tokens=16 scale_factor=18923.812500 estimated_measured_ms=54134378.151 estimated_measured_s=54134.378 measured_tokens_per_s=5.593
bench-file-phases: model_read_ms=1227.084 gguf_parse_ms=8.054 model_load_ms=2379.588 tokenizer_setup_ms=118.976 input_read_ms=2.349 tokenize_ms=1586.070 token_prefix_ms=0.110 warmup_ms=0.000 measured_ms=2860.649 total_ms=14136.766
```

The broader 64-token prefix matrix uses the same harness with `--limit-tokens
64`, `--n-ctx 128`, and `--progress-every 16`:

```sh
scripts/run-target-bench-smoke.sh --input /tmp/enwik8 --tinyllama-q8 /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q8_0.gguf --tinyllama-q4 /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --qwen25-q8 /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf --smollm2-q8 /tmp/detllm-external/SmolLM2-1.7B-Instruct-Q8_0.gguf --limit-tokens 64 --n-ctx 128 --progress-every 16
```

```text
== tinyllama-q8 ==
bench-file-progress phase=encode tokens_done=16 tokens_total=64 elapsed_ms=2080.736 tokens_per_s=7.690
bench-file-progress phase=encode tokens_done=32 tokens_total=64 elapsed_ms=4381.900 tokens_per_s=7.303
bench-file-progress phase=encode tokens_done=48 tokens_total=64 elapsed_ms=6659.421 tokens_per_s=7.208
bench-file-progress phase=encode tokens_done=64 tokens_total=64 elapsed_ms=8966.214 tokens_per_s=7.138
bench-file model=/tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q8_0.gguf input=/tmp/enwik8 limit_bytes=1048576 limit_tokens=64 iters=1 warmup=false mode=encode-only threads=8 n_ctx=128 overlap=32 model_sha256=a4c9bb1dbaa372f6381a035fa5c02ef087aaa1ff1f843a56a22328114f03fc59 input_sha256=f2ef9d4f53049b3642e646fe06024f1b025ce73dcc83a317c0a55eed1004ac56
bench-file: source_input_bytes=100000000 measured_input_bytes=169 total_input_bytes=169 tokenized_tokens=336344 tokens=64 total_tokens=64 payload_bytes=129 dtlz_bytes=185 payload_bits_per_byte=6.106509 dtlz_bits_per_byte=8.757396 compression_ratio=1.094675 elapsed_ms=9023.003 input_bytes_per_s=18.730 tokens_per_s=7.093
bench-file-estimate: full_tokens=336344 full_input_bytes=1048576 measured_tokens=64 scale_factor=5255.375000 estimated_measured_ms=47419262.857 estimated_measured_s=47419.263 measured_tokens_per_s=7.093
bench-file-phases: model_read_ms=2017.435 gguf_parse_ms=6.854 model_load_ms=1395.045 tokenizer_setup_ms=26.968 input_read_ms=52.942 tokenize_ms=924.101 token_prefix_ms=0.128 warmup_ms=0.000 measured_ms=9023.003 total_ms=17325.731
== tinyllama-q4 ==
bench-file-progress phase=encode tokens_done=16 tokens_total=64 elapsed_ms=3075.957 tokens_per_s=5.202
bench-file-progress phase=encode tokens_done=32 tokens_total=64 elapsed_ms=6289.934 tokens_per_s=5.087
bench-file-progress phase=encode tokens_done=48 tokens_total=64 elapsed_ms=9415.671 tokens_per_s=5.098
bench-file-progress phase=encode tokens_done=64 tokens_total=64 elapsed_ms=12579.055 tokens_per_s=5.088
bench-file model=/tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q4_0.gguf input=/tmp/enwik8 limit_bytes=1048576 limit_tokens=64 iters=1 warmup=false mode=encode-only threads=8 n_ctx=128 overlap=32 model_sha256=da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556 input_sha256=f2ef9d4f53049b3642e646fe06024f1b025ce73dcc83a317c0a55eed1004ac56
bench-file: source_input_bytes=100000000 measured_input_bytes=169 total_input_bytes=169 tokenized_tokens=336344 tokens=64 total_tokens=64 payload_bytes=124 dtlz_bytes=180 payload_bits_per_byte=5.869822 dtlz_bits_per_byte=8.520710 compression_ratio=1.065089 elapsed_ms=12607.221 input_bytes_per_s=13.405 tokens_per_s=5.076
bench-file-estimate: full_tokens=336344 full_input_bytes=1048576 measured_tokens=64 scale_factor=5255.375000 estimated_measured_ms=66255676.349 estimated_measured_s=66255.676 measured_tokens_per_s=5.076
bench-file-phases: model_read_ms=590.225 gguf_parse_ms=8.067 model_load_ms=934.405 tokenizer_setup_ms=41.089 input_read_ms=1.960 tokenize_ms=907.220 token_prefix_ms=0.196 warmup_ms=0.000 measured_ms=12607.221 total_ms=17227.674
== qwen25-q8 ==
bench-file-progress phase=encode tokens_done=16 tokens_total=64 elapsed_ms=2888.551 tokens_per_s=5.539
bench-file-progress phase=encode tokens_done=32 tokens_total=64 elapsed_ms=5872.289 tokens_per_s=5.449
bench-file-progress phase=encode tokens_done=48 tokens_total=64 elapsed_ms=8966.890 tokens_per_s=5.353
bench-file-progress phase=encode tokens_done=64 tokens_total=64 elapsed_ms=12027.905 tokens_per_s=5.321
bench-file model=/tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf input=/tmp/enwik8 limit_bytes=1048576 limit_tokens=64 iters=1 warmup=false mode=encode-only threads=8 n_ctx=128 overlap=32 model_sha256=d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8 input_sha256=b4997b129849e53a0cb6265f2561d8e57ad57003ffbcc1c7357b03918e79b03b
bench-file: source_input_bytes=100000000 measured_input_bytes=190 total_input_bytes=190 tokenized_tokens=279472 tokens=64 total_tokens=64 payload_bytes=15 dtlz_bytes=71 payload_bits_per_byte=0.631579 dtlz_bits_per_byte=2.989474 compression_ratio=0.373684 elapsed_ms=12120.284 input_bytes_per_s=15.676 tokens_per_s=5.280
bench-file-estimate: full_tokens=279472 full_input_bytes=1048576 measured_tokens=64 scale_factor=4366.750000 estimated_measured_ms=52926251.153 estimated_measured_s=52926.251 measured_tokens_per_s=5.280
bench-file-phases: model_read_ms=2722.861 gguf_parse_ms=31.691 model_load_ms=2113.508 tokenizer_setup_ms=480.329 input_read_ms=31.406 tokenize_ms=1489.082 token_prefix_ms=0.135 warmup_ms=0.000 measured_ms=12120.284 total_ms=25165.858
== smollm2-q8 ==
bench-file-progress phase=encode tokens_done=16 tokens_total=64 elapsed_ms=2819.196 tokens_per_s=5.675
bench-file-progress phase=encode tokens_done=32 tokens_total=64 elapsed_ms=5766.107 tokens_per_s=5.550
bench-file-progress phase=encode tokens_done=48 tokens_total=64 elapsed_ms=8598.467 tokens_per_s=5.582
bench-file-progress phase=encode tokens_done=64 tokens_total=64 elapsed_ms=11588.129 tokens_per_s=5.523
bench-file model=/tmp/detllm-external/SmolLM2-1.7B-Instruct-Q8_0.gguf input=/tmp/enwik8 limit_bytes=1048576 limit_tokens=64 iters=1 warmup=false mode=encode-only threads=8 n_ctx=128 overlap=32 model_sha256=0f3fb091804c48a561b42a4ca1be9ce2c353017187f74c48f52299cae790abe5 input_sha256=e70374032866c3858fb877a60539f38959a73eb0abbe0417fb58411fe8b5d52a
bench-file: source_input_bytes=100000000 measured_input_bytes=162 total_input_bytes=162 tokenized_tokens=302781 tokens=64 total_tokens=64 payload_bytes=17 dtlz_bytes=73 payload_bits_per_byte=0.839506 dtlz_bits_per_byte=3.604938 compression_ratio=0.450617 elapsed_ms=11678.957 input_bytes_per_s=13.871 tokens_per_s=5.480
bench-file-estimate: full_tokens=302781 full_input_bytes=1048576 measured_tokens=64 scale_factor=4730.953125 estimated_measured_ms=55252597.946 estimated_measured_s=55252.598 measured_tokens_per_s=5.480
bench-file-phases: model_read_ms=1775.138 gguf_parse_ms=10.109 model_load_ms=2204.428 tokenizer_setup_ms=125.453 input_read_ms=8.458 tokenize_ms=1383.989 token_prefix_ms=0.134 warmup_ms=0.000 measured_ms=11678.957 total_ms=23142.203
```

This is target-model throughput and prefix compression smoke evidence for the
current GGUF matrix. Because `--limit-tokens 64` is set, it is not the final
M4 enwik8 first-1MB compression-rate acceptance measurement.

Production-shape Qwen2.5 Q8_0 prefix round-trip with `n_ctx=2048`:

```sh
scripts/run-target-full-bench.sh --model /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf --input /tmp/enwik8 --name qwen25-q8-first1m-c512-roundtrip --limit-tokens 512 --n-ctx 2048 --threads 8 --progress-every 64
```

Completed locally on 2026-07-10. The wrapper wrote the combined progress log
to `/tmp/detllm-target-bench/qwen25-q8-first1m-c512-roundtrip.log` and the
stable summary to
`/tmp/detllm-target-bench/qwen25-q8-first1m-c512-roundtrip.summary`.

```text
bench-file-progress phase=encode tokens_done=512 tokens_total=512 elapsed_ms=99892.145 tokens_per_s=5.126 remaining_s=0.000 estimated_total_s=99.892
bench-file-progress phase=decode tokens_done=512 tokens_total=512 elapsed_ms=100346.083 tokens_per_s=5.102 remaining_s=0.000 estimated_total_s=100.346
bench-file model=/tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf input=/tmp/enwik8 limit_bytes=1048576 limit_tokens=512 iters=1 warmup=false mode=round-trip threads=8 n_ctx=2048 overlap=512 model_sha256=d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8 input_sha256=00474457a0a2b7dab617ddacdab2d0b84ae7de61080c41c57aea1250c7a8413a
bench-file: source_input_bytes=100000000 measured_input_bytes=1702 total_input_bytes=1702 tokenized_tokens=279472 tokens=512 total_tokens=512 payload_bytes=83 dtlz_bytes=139 payload_bits_per_byte=0.390129 dtlz_bits_per_byte=0.653349 compression_ratio=0.081669 elapsed_ms=200446.456 input_bytes_per_s=8.491 tokens_per_s=2.554
bench-file-phases: model_read_ms=3148.833 gguf_parse_ms=26.912 model_load_ms=2167.147 tokenizer_setup_ms=372.169 input_read_ms=31.223 tokenize_ms=1497.431 token_prefix_ms=0.246 warmup_ms=0.000 measured_ms=200446.456 total_ms=213782.146
```

This run uses the final wrapper's production context shape and default
round-trip verification, so it proves byte equality for a larger first-1MB
prefix than the 64-token encode-only matrix. Because `--limit-tokens 512` is
still set, it remains prefix evidence rather than the final full-token M4
compression-rate measurement.

The same production-shape Qwen2.5 Q8_0 512-token round-trip was also run with
`--threads 8` and `--threads 16`:

```sh
scripts/run-target-full-bench.sh --model /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf --input /tmp/enwik8 --name qwen25-q8-thread8-512rt --limit-tokens 512 --n-ctx 2048 --threads 8 --progress-every 512
scripts/run-target-full-bench.sh --model /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf --input /tmp/enwik8 --name qwen25-q8-thread16-512rt --limit-tokens 512 --n-ctx 2048 --threads 16 --progress-every 512
```

Observed output:

| threads | measured bytes | payload bytes | DTLZ bytes | DTLZ SHA-256 | round-trip throughput |
|---:|---:|---:|---:|---|---:|
| 8 | 1702 | 83 | 139 | `8eb550073f2296b34c38a3192c93adb1a8c41245d08048fc812fd98d938f0ab7` | 2.522 tokens/s |
| 16 | 1702 | 83 | 139 | `8eb550073f2296b34c38a3192c93adb1a8c41245d08048fc812fd98d938f0ab7` | 1.710 tokens/s |

Both runs used `mode=round-trip`, `n_ctx=2048`, `overlap=512`, and the same
measured input SHA-256
`00474457a0a2b7dab617ddacdab2d0b84ae7de61080c41c57aea1250c7a8413a`.
This is target-model evidence that the production-shape codec payload is
thread-count invariant for this larger prefix, though it still remains
token-limited preflight evidence rather than the final full first-1MB run.

Production-shape Qwen2.5 Q8_0 encode-only preflight over one full `n_ctx=2048`
measured prefix:

```sh
scripts/run-target-full-bench.sh --model /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf --input /tmp/enwik8 --name qwen25-q8-first1m-c2048-encode --limit-tokens 2048 --n-ctx 2048 --threads 8 --progress-every 256 --encode-only --estimate-full-run
```

Completed locally on 2026-07-10. The wrapper wrote the combined progress log
to `/tmp/detllm-target-bench/qwen25-q8-first1m-c2048-encode.log`, the stable
summary to `/tmp/detllm-target-bench/qwen25-q8-first1m-c2048-encode.summary`,
and the latest progress row to
`/tmp/detllm-target-bench/qwen25-q8-first1m-c2048-encode.progress`.

```text
bench-file-progress phase=encode tokens_done=256 tokens_total=2048 elapsed_ms=50928.032 tokens_per_s=5.027 remaining_s=356.496 estimated_total_s=407.424
bench-file-progress phase=encode tokens_done=512 tokens_total=2048 elapsed_ms=102578.300 tokens_per_s=4.991 remaining_s=307.735 estimated_total_s=410.313
bench-file-progress phase=encode tokens_done=768 tokens_total=2048 elapsed_ms=155243.154 tokens_per_s=4.947 remaining_s=258.739 estimated_total_s=413.982
bench-file-progress phase=encode tokens_done=1024 tokens_total=2048 elapsed_ms=209195.353 tokens_per_s=4.895 remaining_s=209.195 estimated_total_s=418.391
bench-file-progress phase=encode tokens_done=1280 tokens_total=2048 elapsed_ms=264427.643 tokens_per_s=4.841 remaining_s=158.657 estimated_total_s=423.084
bench-file-progress phase=encode tokens_done=1536 tokens_total=2048 elapsed_ms=320505.183 tokens_per_s=4.792 remaining_s=106.835 estimated_total_s=427.340
bench-file-progress phase=encode tokens_done=1792 tokens_total=2048 elapsed_ms=378014.076 tokens_per_s=4.741 remaining_s=54.002 estimated_total_s=432.016
bench-file-progress phase=encode tokens_done=2048 tokens_total=2048 elapsed_ms=436779.620 tokens_per_s=4.689 remaining_s=0.000 estimated_total_s=436.780
bench-file model=/tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf input=/tmp/enwik8 limit_bytes=1048576 limit_tokens=2048 iters=1 warmup=false mode=encode-only threads=8 n_ctx=2048 overlap=512 model_sha256=d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8 input_sha256=2254bc36b0d368e41115ed9ff9dcf77b3057c28b2330dab7db3667cb196a7966
bench-file: source_input_bytes=100000000 measured_input_bytes=6748 total_input_bytes=6748 tokenized_tokens=279472 tokens=2048 total_tokens=2048 payload_bytes=481 dtlz_bytes=537 payload_bits_per_byte=0.570243 dtlz_bits_per_byte=0.636633 compression_ratio=0.079579 elapsed_ms=436906.749 input_bytes_per_s=15.445 tokens_per_s=4.687
bench-file-estimate: full_tokens=279472 full_input_bytes=1048576 measured_tokens=2048 scale_factor=136.460938 estimated_measured_ms=59620704.567 estimated_measured_s=59620.705 measured_tokens_per_s=4.687
bench-file-phases: model_read_ms=2193.246 gguf_parse_ms=25.645 model_load_ms=2184.654 tokenizer_setup_ms=469.556 input_read_ms=30.966 tokenize_ms=1575.105 token_prefix_ms=0.445 warmup_ms=0.001 measured_ms=436906.749 total_ms=449697.079
```

This run measures the same production context length as the final wrapper
default and covers a complete 2048-token window. Because it uses
`--encode-only` and `--limit-tokens 2048`, it remains a long-prefix preflight:
it does not replace the final no-token-limit round-trip M4 acceptance
measurement.

Current-format Qwen2.5 preflight with a one-token measured prefix records the
full first-1MB tokenization size:

```sh
cargo run --release -p xtask --features parallel,simd -- bench-file --model /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf --input /tmp/enwik8 --limit-bytes 1048576 --limit-tokens 1 --n-ctx 8 --threads 8 --iters 1 --no-warmup --show-phases
```

```text
bench-file model=/tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf input=/tmp/enwik8 limit_bytes=1048576 limit_tokens=1 iters=1 warmup=false threads=8 n_ctx=8 overlap=2 model_sha256=d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8 input_sha256=dabd3aff769f07eb2965401eb029974ebba3407afd02b26ddb564ea5f8efae72
bench-file: source_input_bytes=100000000 measured_input_bytes=1 total_input_bytes=1 tokenized_tokens=279472 tokens=1 total_tokens=1 payload_bytes=9 dtlz_bytes=65 payload_bits_per_byte=72.000000 dtlz_bits_per_byte=520.000000 compression_ratio=65.000000 elapsed_ms=466.371 input_bytes_per_s=2.144 tokens_per_s=2.144
bench-file-phases: model_read_ms=2108.649 gguf_parse_ms=27.972 model_load_ms=2488.917 tokenizer_setup_ms=367.869 input_read_ms=10.851 tokenize_ms=1194.354 token_prefix_ms=0.109 warmup_ms=0.000 measured_ms=466.371 total_ms=12709.625
```

This is input-scale and round-trip evidence for the `bench-file`
implementation on the canonical enwik8 byte stream, not a meaningful language
model compression-quality result. The tiny fixture has byte tokens and a tiny
context, so it is expected to produce near-raw 8 bpb output.

`bench-file` tokenizes the input and encodes the token stream on every
measured iteration. In the default round-trip mode it also decodes and
detokenizes back to bytes on every measured iteration; in `--encode-only` mode
it stops after payload generation. The codec path keeps a streaming KV cache
within each fixed window and replays only the configured overlap when a window
rolls over; it does not rebuild the full prefix CDF for every token. The tests
`streaming_codec_matches_replay_cdf_payload` and
`xtask_streaming_codec_matches_replay_cdf_payload` compare the streaming
payload against the direct replay rule byte-for-byte. The harness reports
payload size and DTLZ size, including the 56-byte file header. It also reports
model and measured input SHA-256 values, source and measured input byte counts,
the tokenized token count before `--limit-tokens`, one-iteration and total
measured token counts, payload-only bpb, DTLZ bpb, compression ratio, elapsed
time, bytes/s, tokens/s, whether a pre-measurement warmup round-trip or
encode-only warmup was run, the measurement mode, and the thread override used
for model kernels.
`--limit-bytes N` truncates the input to at most the first `N` bytes before
tokenization, and the harness reads only that prefix while retaining the full
source file size in `source_input_bytes`. The enwik8 first-1MB measurement can
therefore use `--limit-bytes 1048576` without creating a separate file or
reading all 100MB into memory. `--limit-tokens N` then truncates the tokenized
stream and detokenizes that prefix back to bytes before measurement; this gives
a reproducible target-model prefix smoke path for long runs while keeping the
reported byte counts and SHA-256 tied to the actual bytes round-tripped.
Tokenization still happens before token truncation.
The ByteBPE path uses a priority-queue merge implementation, so 1MB byte caps
are usable for Qwen2.5 prefix preflights; use smaller `--limit-bytes` values
only when an even faster smoke is needed. Omit `--limit-tokens` for the final
first-1MB acceptance measurement. `--threads N` fixes the model parallelism for
reproducible benchmark notes, and `--no-warmup` skips the extra
pre-measurement pass for long target-model measurements. In round-trip mode,
the measured iteration still verifies encode/decode byte round-trip; in
encode-only mode, use a separate short round-trip smoke for codec correctness.
`--show-phases` adds an opt-in `bench-file-phases` line for model
read/parse/load, tokenizer setup, input read, tokenization, token-prefix
detokenization, warmup, measured loop, and total wall time.
`--summary PATH` writes the final stdout summary lines to a file through a
same-directory temporary file and rename, which is useful for long target-model
runs where progress output is noisy or the terminal history is transient.
`--output-dtlz PATH` similarly writes the measured iteration's DTLZ file
through a temporary file and rename as soon as encode finishes. It requires
`--iters 1`, matching the target-model acceptance run, so a long round-trip
run leaves a durable compressed file before the decode verification phase
starts. That file can be checked later with the public `decompress` command.
`--checkpoint PATH --checkpoint-every N` can be combined with `--output-dtlz`
on the same single-iteration shape. It atomically saves the range encoder
state and completed codec-symbol count during encode. Rerunning the same
command resumes from that checkpoint after validating the model SHA-256, input
SHA-256, context settings, measured input length, and token count. The
checkpoint is removed only after the final DTLZ has been written and any
round-trip verification has completed.
`--verify-dtlz PATH` resumes from such a saved DTLZ file: it reloads the same
model and input prefix, verifies the DTLZ header model SHA/context/original
length, decodes the payload, compares the decoded codec symbols with the fresh
tokenization, and detokenizes back to the measured input bytes. This mode is
intended for long target-model runs where encode completed and wrote the DTLZ
but the decode/round-trip verification still needs to be repeated.
For the final target-model first-1MB run, use:

```sh
scripts/run-target-full-bench.sh --model /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf --input /tmp/enwik8 --name qwen25-q8-first1m
```

If that run has already written `/tmp/detllm-target-bench/qwen25-q8-first1m.dtlz`
and only the verification phase needs to be resumed, use:

```sh
scripts/run-target-full-bench.sh --model /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf --input /tmp/enwik8 --name qwen25-q8-first1m --verify-dtlz /tmp/detllm-target-bench/qwen25-q8-first1m.dtlz
```

The wrapper keeps the acceptance defaults explicit: `--limit-bytes 1048576`,
no `--limit-tokens`, round-trip mode, `--no-warmup`, `--show-phases`,
`--threads 8`, `--n-ctx 2048`, and `--progress-every 1000`. It records the
combined progress log, the stable `bench-file --summary` output, and a
`<name>.dtlz` output in `/tmp/detllm-target-bench` unless `--out DIR` is
provided. For encode runs it also passes `--checkpoint
/tmp/detllm-target-bench/<name>.checkpoint --checkpoint-every 1000`, so an
interrupted encode can be rerun without discarding completed range-coder work.
In `--verify-dtlz` mode the same wrapper records summary/progress/log files
for the resumed decode verification without rewriting the saved DTLZ. It keeps
a `<name>.progress` file updated with the latest progress row. Use
`--limit-tokens N --encode-only --estimate-full-run` with the same wrapper only
for preflight estimates; omit those flags for the acceptance measurement.
The `check-helper-scripts` hygiene check also validates the
`scripts/run-target-full-bench.sh` wrapper shape: the first-1MB byte limit,
production context/thread defaults, summary/progress outputs, DTLZ output,
checkpoint output, `--verify-dtlz` branch, and no-warmup default must remain
present. The unit test
`target_full_bench_script_check_requires_resume_safe_shape` verifies that the
check fails if the checkpoint, progress-summary option, or verify-mode guards
are removed.
The tiny-fixture unit test `bench_file_verify_dtlz_replays_saved_output`
exercises the same bench-file DTLZ persistence and resume-verification path
without external models: it writes a DTLZ plus summary through `--output-dtlz`,
asserts the checkpoint is removed after completion, then reruns through
`--verify-dtlz` and checks the verify summary records `mode=verify-dtlz`, the
`bench-file-verify-dtlz` row, and the restored SHA-256. Local validation on
commit `18d4ae3` passed:

```sh
cargo fmt --all --check
cargo test -p xtask
cargo clippy -p xtask --all-targets -- -D warnings
```

GitHub Actions run
<https://github.com/mii443/detllm/actions/runs/29084830217> completed
successfully for the same commit, including the `hygiene`,
native/wasm/toolchain jobs, and final `logits-hash-match` artifact comparison.
The follow-up documentation commit `684335e` also completed successfully in
GitHub Actions run
<https://github.com/mii443/detllm/actions/runs/29085088359>, covering the same
push CI matrix after recording this DTLZ verification coverage.
`--estimate-full-run` adds an opt-in `bench-file-estimate` line for
`--limit-tokens` preflights, reporting the full tokenized prefix, full input
byte count, scale factor, measured token throughput, and estimated measured
loop time for running without the token limit.
`--progress-every N` emits `bench-file-progress` lines on stderr every N encode
or decode tokens and at phase completion; progress lines include elapsed time,
tokens/s, remaining seconds, and estimated total seconds for the current
phase. The stdout summary lines remain stable for copying into this file. The
Qwen2.5 prefix run above shows 1MB
ByteBPE tokenization is about 1.5 seconds on this host; the estimate line shows
the current full 279,472-token measured encode loop is roughly 16.6 hours when
estimated from the 2048-token production-context preflight. After streaming
KV-cache reuse and validated-model hot-path checks, the current 64-token
encode-only measured loop is roughly 12 seconds and the 2048-token measured
loop is roughly 437 seconds. The model forward path also reuses
`ForwardWorkspace` scratch buffers across tokens, avoiding per-token allocation
of the large hidden-state, projection, attention, feed-forward, Q8A, and Q8_K
activation buffers, and uses layout checks for already-loaded models instead
of re-scanning all weight tensors on every token and GEMV. Attention reads
KV-cache prefix slices directly instead of copying per-head key/value windows.
Codec encode computes only the selected symbol's range-coder interval from the
logits and no longer materializes full frequency/cumulative CDF vectors per
token; tests verify this symbol-range API matches the full CDF exactly,
including byte-escape tails. Decode uses a frequency-only distribution: it
computes the exact range-coder total without building the cumulative vector,
then scans the frequency slice for the decoded value. Tests verify this decoder
distribution returns the same symbol ranges as the full CDF, including
byte-escape tails, while the public validating `symbol_for` helper remains
available for untrusted tables. The tests also verify the scratch API is
bit-for-bit equivalent to the owned `logits_to_cdf` API and that streaming
codec payloads still match the direct replay rule.
`logits_to_cdf_clamps_exp_input_lower_bound` fixes the softmax lower clamp:
logits far below the maximum, such as `-1000.0`, must produce the same CDF and
byte-escape symbol ranges as the specified `-88.0` clamp. With the `parallel`
feature,
row-parallel GEMV reuses fixed-size Rayon worker pools keyed by `--threads`
instead of spawning OS threads for every matrix multiply, attention
parallelizes independent heads with per-head score/prob scratch for larger
attention windows while keeping each head's softmax and value accumulation
ordered, and CDF construction parallelizes only the independent `exp[i]` fill
while keeping `Z` and prefix sums single-threaded.
This is the harness to use for target-model enwik8 first-1MB measurements; the
bundled fixtures remain smoke and input-scale checks.
The harness applies the same tokenizer/model vocabulary equality check and
`2^18` codec vocabulary bound as the CLI compression path before accepting a
model for measurement.
After the decode-side frequency-only CDF path, the following target-model smoke
validated the saved DTLZ output with the public decompressor:

```sh
scripts/run-target-full-bench.sh \
  --model /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf \
  --input /tmp/enwik8 \
  --name qwen25-q8-cdf-decode-smoke \
  --limit-tokens 64 \
  --n-ctx 128 \
  --progress-every 16
cargo run --release -p det-cli --features parallel,simd -- decompress \
  -m /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf \
  -i /tmp/detllm-target-bench/qwen25-q8-cdf-decode-smoke.dtlz \
  -o /tmp/detllm-target-bench/qwen25-q8-cdf-decode-smoke.restored \
  --threads 8
sha256sum /tmp/detllm-target-bench/qwen25-q8-cdf-decode-smoke.dtlz \
  /tmp/detllm-target-bench/qwen25-q8-cdf-decode-smoke.restored
```

Observed output:

```text
bench-file: source_input_bytes=100000000 measured_input_bytes=190 total_input_bytes=190 tokenized_tokens=279472 tokens=64 total_tokens=64 payload_bytes=15 dtlz_bytes=71 payload_bits_per_byte=0.631579 dtlz_bits_per_byte=2.989474 compression_ratio=0.373684 elapsed_ms=24599.368 input_bytes_per_s=7.724 tokens_per_s=2.602
bench-file-output-dtlz path=/tmp/detllm-target-bench/qwen25-q8-cdf-decode-smoke.dtlz bytes=71 sha256=eab211252eb7c9af0d50ed29e0f14e5876a8de69b167505ae0807ae217a25b43
eab211252eb7c9af0d50ed29e0f14e5876a8de69b167505ae0807ae217a25b43  /tmp/detllm-target-bench/qwen25-q8-cdf-decode-smoke.dtlz
b4997b129849e53a0cb6265f2561d8e57ad57003ffbcc1c7357b03918e79b03b  /tmp/detllm-target-bench/qwen25-q8-cdf-decode-smoke.restored
```

The unit test `xtask_codec_helpers_reject_invalid_windows` also covers the
lower-level bench codec helpers directly: their encode/decode paths reject
zero `n_ctx`, `overlap >= n_ctx`, and `n_ctx` larger than the model context
before any benchmark token stream is processed.

## Fixture Benchmark Snapshot

Command:

```sh
cargo run --release -p xtask -- bench-testdata --iters 100
```

Environment:

```text
Linux main-win 6.6.87.2-microsoft-standard-WSL2 #1 SMP PREEMPT_DYNAMIC Thu Jun 5 18:30:46 UTC 2025 x86_64
CPU: AMD Ryzen 9 7950X3D 16-Core Processor, 32 logical CPUs
Commit: 52288f1
rustc 1.95.0 (59807616e 2026-04-14)
```

Observed output:

```text
bench-testdata iters=100
logits tiny-f32: hash=92a0280149c6b1505c84dce0d19486a2093f93b7978b579c220000d12e4ef7e7 tokens=600 elapsed_ms=4.446 tokens_per_s=134939.412
logits tiny-qmix: hash=8a34d3c4a05e9a30b90aadcdca7b6bac91655e6ab67980ccdb6726565d35f3e4 tokens=600 elapsed_ms=4.743 tokens_per_s=126505.468
codec tiny-f32: input_bytes=3900 payload_bytes=4600 elapsed_ms=43.192 input_bytes_per_s=90294.961
codec tiny-qmix: input_bytes=3900 payload_bytes=4600 elapsed_ms=49.637 input_bytes_per_s=78570.804
```

`bench-testdata` verifies that the fixture logits hash does not change during
the measured loop, and each codec benchmark decodes the measured payload and
checks byte equality. This is an equivalent harness result for the bundled
fixtures only; target-model benchmark results remain separate acceptance
evidence.

The manual GitHub Actions `benchmarks.yml` workflow collects the same
`bench-testdata` fixture benchmark on hosted `x86_64-linux`, `aarch64-linux`,
and `aarch64-macos` runners without adding benchmark timing noise to push/PR
CI:

```sh
gh workflow run benchmarks.yml --repo mii443/detllm --ref main -f iters=100
```

Each matrix job writes runner OS/architecture, `uname -a`, `rustc --version`,
the exact command, and the benchmark output to a `bench-testdata-*` artifact.

Completed GitHub Actions benchmark evidence:

- Repository: `mii443/detllm`
- Commit: `e6136d8c9c392f84d46b53d56310399cdf15c205`
  (`Add manual benchmark workflow`)
- Run: <https://github.com/mii443/detllm/actions/runs/29050786923>
- Result: passed
- Command: `cargo run --release -p xtask -- bench-testdata --iters 100`
- rustc: `rustc 1.97.0 (2d8144b78 2026-07-07)`

Observed hosted matrix output:

| target | runner | kernel | `tiny-f32` logits | `tiny-qmix` logits | `tiny-f32` codec | `tiny-qmix` codec |
|---|---|---|---:|---:|---:|---:|
| `x86_64-linux` | Linux X64 | Linux 6.17.0-1018-azure | 82645.971 tokens/s | 80591.466 tokens/s | 55868.648 input bytes/s | 55616.437 input bytes/s |
| `aarch64-linux` | Linux ARM64 | Linux 6.17.0-1018-azure | 119452.311 tokens/s | 101791.963 tokens/s | 75761.730 input bytes/s | 67332.095 input bytes/s |
| `aarch64-macos` | macOS ARM64 | Darwin 23.6.0 | 132672.418 tokens/s | 105609.799 tokens/s | 167062.304 input bytes/s | 90900.173 input bytes/s |

All three artifacts reported the same fixture hashes:

- `tiny-f32`: `92a0280149c6b1505c84dce0d19486a2093f93b7978b579c220000d12e4ef7e7`
- `tiny-qmix`: `8a34d3c4a05e9a30b90aadcdca7b6bac91655e6ab67980ccdb6726565d35f3e4`

## Logits Cosine Compare Harness

Command:

```sh
cargo run -p det-cli -- tokenize -m model.gguf -p "prompt text"
scripts/reference_logits_transformers.py --model-id HF_MODEL_ID --tokens TOKENS --out reference.logits.bin --expected-rows TOKENS --expected-vocab VOCAB
cargo run -p xtask -- compare-logits --actual detllm.logits.bin --reference reference.logits.bin --row-size VOCAB --rows TOKENS --min-cosine 0.999 --worst-rows 3 --top-diffs 10
```

`detllm tokenize` emits the comma-separated token IDs produced by the same
GGUF tokenizer path used by `logits -p`, after checking that tokenizer and
model vocabulary lengths match. These IDs are the prompt-shape evidence to
record next to an external HF transformers or llama.cpp reference dump.
`compare-logits` reads two little-endian f32 logits dumps, rejects malformed
lengths and non-finite values, then reports global cosine similarity, maximum
absolute difference, and RMS difference. With `--row-size`, it also reports
the minimum row index and cosine across token positions and applies
`--min-cosine` to both the global cosine and minimum row cosine. With
`--rows`, it rejects dumps whose row count does not match the expected number
of token positions, which prevents comparing different prompt lengths or a
single final-token dump against a full-position detllm dump. `--worst-rows`
prints the lowest-cosine rows, and `--top-diffs` prints the largest absolute
logit differences with row/column coordinates when `--row-size` is present.
For quantized target GGUFs, the acceptance raw-logits cosine gate is the
same-GGUF llama.cpp comparison at `--min-cosine 0.999`. HF Transformers dumps
from the original f32 model are still useful diagnostics, but they are not the
fixed-threshold acceptance reference for Q8_0/Q4_0 GGUFs unless a
model-specific quantization-aware threshold is deliberately chosen.

The helper script
[`scripts/reference_logits_transformers.py`](../scripts/reference_logits_transformers.py)
generates that reference dump from an HF Transformers causal LM without adding
Python dependencies to the Rust workspace. It accepts either explicit token IDs
or a prompt; for external validation, prefer explicit IDs produced by
`detllm tokenize` so both systems evaluate the same token stream. The script
writes the same row-major little-endian f32 format as `detllm logits --dump`
and can enforce the expected row count and vocabulary size before writing the
file.

The helper program
[`scripts/reference_logits_llamacpp.cpp`](../scripts/reference_logits_llamacpp.cpp)
generates the same raw row-major little-endian f32 dump through the llama.cpp C
API when local `llama.h` and `libllama` installations are available. It accepts
explicit token IDs only, marks every token position for logits output, can
decode one token at a time with `--sequential` to mirror detllm's streaming
forward path, can force llama.cpp K/V cache tensors to F32 with `--kv-f32` for
diagnostics, and can enforce row count and vocabulary size before writing the
file.

Example TinyLlama command shape:

```sh
cargo run --release -p det-cli -- logits -m /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q8_0.gguf --tokens 1,2,3 --dump detllm.logits.bin --hash --threads 8
scripts/reference_logits_transformers.py --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tokens 1,2,3 --out reference.logits.bin --expected-rows 3 --expected-vocab 32000 --threads 1 --dtype float32
cargo run -p xtask -- compare-logits --actual detllm.logits.bin --reference reference.logits.bin --row-size 32000 --rows 3 --min-cosine 0.999
```

Reproducible target-model HF Transformers wrapper:

```sh
scripts/run-target-hf-logits-matrix.sh \
  --tinyllama-q8 /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  --tinyllama-q4 /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  --qwen25-q8 /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf \
  --smollm2-q8 /tmp/detllm-external/SmolLM2-1.7B-Instruct-Q8_0.gguf \
  --out /tmp/detllm-hf-logits-matrix-20260710 \
  --threads 8 \
  --torch-threads 1 \
  --dtype float32
```

The wrapper runs the same short low-level token streams and tokenizer-backed
8-token validation streams used by the llama.cpp raw-logits broad matrix:
TinyLlama Q8_0/Q4_0 against
`TinyLlama/TinyLlama-1.1B-Chat-v1.0`, Qwen2.5 Q8_0 against
`Qwen/Qwen2.5-1.5B-Instruct`, and SmolLM2 Q8_0 against
`HuggingFaceTB/SmolLM2-1.7B-Instruct`. Each row writes a detllm dump, an HF
Transformers dump, and a `compare-logits` report under `--out`. The HF model
arguments can also point at local model directories for offline validation.
The wrapper defaults to `--min-cosine 0.0` so quantized-GGUF-vs-HF-original
comparisons record every row. Pass an explicit threshold, such as
`--min-cosine 0.999`, only when comparing an equivalent f32 path or a calibrated
model-specific threshold.

The script's syntax and argument parser were checked locally without writing
bytecode with:

```sh
python3 -B scripts/reference_logits_transformers.py --help
bash -n scripts/run-target-hf-logits-matrix.sh
scripts/run-target-hf-logits-matrix.sh --help
```

The HF matrix was run on 2026-07-10 on the local host with `torch 2.7.1+cu126`,
`transformers 5.13.0`, `safetensors 0.8.0`, `sentencepiece 0.2.1`,
`scipy 1.15.3`, `--dtype float32`, `--torch-threads 1`, and detllm commit
`2ea2d3d`. The command used `--min-cosine 0.0` so all rows would be recorded
even when the design's `0.999` target was not met:

```sh
scripts/run-target-hf-logits-matrix.sh \
  --tinyllama-q8 /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  --tinyllama-q4 /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  --qwen25-q8 /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf \
  --smollm2-q8 /tmp/detllm-external/SmolLM2-1.7B-Instruct-Q8_0.gguf \
  --out /tmp/detllm-hf-logits-matrix-20260710 \
  --threads 8 \
  --torch-threads 1 \
  --dtype float32 \
  --min-cosine 0.0
```

Observed HF f32-original comparison results:

| model | case | rows | vocab | overall cosine | min row cosine | max abs diff | rms diff | status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| TinyLlama Q8_0 | `ids-1-2-3` | 3 | 32000 | 0.998597893 | 0.997304256 | 1.150591612 | 0.157622818 | below `0.999` |
| TinyLlama Q8_0 | `hello-validation-8` | 8 | 32000 | 0.999140071 | 0.996110549 | 2.072637081 | 0.189306373 | below `0.999` |
| TinyLlama Q4_0 | `ids-1-2-3` | 3 | 32000 | 0.888001168 | 0.805630281 | 13.091418743 | 1.390802840 | below `0.999` |
| TinyLlama Q4_0 | `hello-validation-8` | 8 | 32000 | 0.930995510 | 0.689283705 | 13.645533442 | 1.648519377 | below `0.999` |
| Qwen2.5 Q8_0 | `special-hello-special` | 3 | 151936 | 0.992294342 | 0.987721191 | 2.586650014 | 0.465782622 | below `0.999` |
| Qwen2.5 Q8_0 | `hello-validation-8` | 8 | 151936 | 0.996246571 | 0.980463056 | 2.738610268 | 0.300313601 | below `0.999` |
| SmolLM2 Q8_0 | `ids-1-2-3` | 3 | 49152 | 0.974184954 | 0.955967579 | 5.795506358 | 1.019470888 | below `0.999` |
| SmolLM2 Q8_0 | `hello-validation-8` | 8 | 49152 | 0.986475034 | 0.953664992 | 8.137038231 | 0.536524322 | below `0.999` |

These results are independent HF evidence, but they are not acceptance-pass
evidence for the `0.999` threshold. They compare quantized target GGUF inference
against the original HF f32 models, while the same-GGUF llama.cpp raw-logits
matrix above remains the passing reference check for implementation parity.
The acceptance decision is to keep the quantized-model raw-logits gate on that
same-GGUF llama.cpp comparison. HF-original comparisons remain diagnostic
negative evidence unless a future model-specific quantization-aware threshold is
added deliberately.

TinyLlama Q8_0 llama.cpp raw-logits reference command:

```sh
c++ -std=c++17 -O2 -I/usr/local/include scripts/reference_logits_llamacpp.cpp -L/usr/local/lib -Wl,-rpath,/usr/local/lib -lllama -lggml -lggml-cpu -lggml-base -o /tmp/reference_logits_llamacpp
cargo run --release -p det-cli -- logits -m /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q8_0.gguf --tokens 1,2,3 --dump /tmp/detllm-tiny-123.rawlogits.bin --hash --threads 8
/tmp/reference_logits_llamacpp --model /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q8_0.gguf --tokens 1,2,3 --out /tmp/llamacpp-tiny-123.rawlogits.bin --threads 8 --ctx-size 16 --batch-size 16 --expected-vocab 32000 --expected-rows 3 --quiet
cargo run -p xtask -- compare-logits --actual /tmp/detllm-tiny-123.rawlogits.bin --reference /tmp/llamacpp-tiny-123.rawlogits.bin --row-size 32000 --rows 3 --min-cosine 0.999
```

Observed output:

```text
79600ae16f6ba067de254839a0df605a1082b2eb6f75b538411be9403fe9251c
reference_logits_llamacpp rows=3 vocab=32000 values=96000
compare-logits values=96000 cosine=0.999815074 max_abs_diff=0.364835739 rms_diff=0.057082669 rows=3 row_size=32000 min_row_cosine=0.999729601
```

TinyLlama Q4_0 llama.cpp raw-logits reference command:

```sh
cargo run --release -p det-cli -- logits -m /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --tokens 1,2,3 --dump /tmp/detllm-tinyllama-q4-123.rawlogits.bin --hash --threads 8
/tmp/reference_logits_llamacpp --model /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --tokens 1,2,3 --out /tmp/llamacpp-tinyllama-q4-123.rawlogits.bin --threads 8 --ctx-size 16 --batch-size 16 --expected-vocab 32000 --expected-rows 3 --quiet
cargo run --release -p xtask -- compare-logits --actual /tmp/detllm-tinyllama-q4-123.rawlogits.bin --reference /tmp/llamacpp-tinyllama-q4-123.rawlogits.bin --row-size 32000 --rows 3 --min-cosine 0.999
```

Observed output:

```text
450bf34ee63249f042cde2156643a53261034a4fa04bf47721da9d865ada9251
reference_logits_llamacpp rows=3 vocab=32000 values=96000
compare-logits values=96000 cosine=0.999667056 max_abs_diff=0.416365802 rms_diff=0.064927172 rows=3 row_size=32000 min_row=2 min_row_cosine=0.999624876
```

TinyLlama Q8_0 longer 8-token text raw-logits reference command:

```sh
cargo run --release -p det-cli -- tokenize -m /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q8_0.gguf -p "Hello world from detllm validation."
cargo run --release -p det-cli -- logits -m /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q8_0.gguf --tokens 10994,3186,515,1439,645,112,8845,49 --dump /tmp/detllm-tiny-hello-validation-8.rawlogits.bin --hash --threads 8
/tmp/reference_logits_llamacpp --model /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q8_0.gguf --tokens 10994,3186,515,1439,645,112,8845,49 --out /tmp/llamacpp-tiny-hello-validation-8.rawlogits.bin --threads 8 --ctx-size 16 --batch-size 16 --expected-vocab 32000 --expected-rows 8 --quiet
cargo run -p xtask -- compare-logits --actual /tmp/detllm-tiny-hello-validation-8.rawlogits.bin --reference /tmp/llamacpp-tiny-hello-validation-8.rawlogits.bin --row-size 32000 --rows 8 --min-cosine 0.999
```

Observed output:

```text
tokens("Hello world from detllm validation.") = 10994,3186,515,1439,645,112,8845,49
d8788e1c61337805f246908b0ccefbbd7ce98d41bb6d0a5efbd98fa6f10f7c12
reference_logits_llamacpp rows=8 vocab=32000 values=256000
compare-logits values=256000 cosine=0.999917601 max_abs_diff=0.386859894 rms_diff=0.057615436 rows=8 row_size=32000 min_row_cosine=0.999801524
```

llama.cpp `llama-perplexity --save-all-logits` does not write the same raw f32
logits matrix as `detllm logits --dump`; the file starts with `_logits_` and
stores the evaluated token log-probability distributions in the quantized
format used by llama.cpp's KL-divergence path. The source implementation is
`tools/perplexity/perplexity.cpp` in llama.cpp. `xtask
compare-llamacpp-logprobs` parses that format, mirrors llama.cpp's per-chunk
BOS handling, converts detllm logits to log-probabilities, and reports both
full-distribution and target-token differences. Because llama.cpp clips the
saved distribution tail to a 16-nat band before uint16 encoding, the target
token metric is the thresholded reference check for perplexity-path parity.

The reproducible target-model wrapper is:

```sh
scripts/run-target-logprob-matrix.sh --tinyllama-q8 /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q8_0.gguf --tinyllama-q4 /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --qwen25-q8 /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf --smollm2-q8 /tmp/detllm-external/SmolLM2-1.7B-Instruct-Q8_0.gguf
```

It runs `llama-perplexity --save-all-logits` for each model with the same
two-chunk validation prompt used below, then checks each dump with `xtask
compare-llamacpp-logprobs --max-target-abs-diff 0.2`. This makes the
perplexity-path reference-quality evidence reproducible in the same style as
the raw-logits, round-trip, logits determinism, codec determinism, and
bench-file target-model matrices.

Observed scripted matrix output:

```text
== tinyllama-q8 ==
compare-llamacpp-logprobs chunks=2 n_ctx=8 vocab=32000 rows=6 values=192000 add_bos=true bos_token=1 max_abs_diff=10.164945602 rms_diff=0.897793605 max_target_abs_diff=0.046883583
== tinyllama-q4 ==
compare-llamacpp-logprobs chunks=2 n_ctx=8 vocab=32000 rows=6 values=192000 add_bos=true bos_token=1 max_abs_diff=11.470438004 rms_diff=1.231016547 max_target_abs_diff=0.110750198
== qwen25-q8 ==
compare-llamacpp-logprobs chunks=2 n_ctx=8 vocab=151936 rows=6 values=911616 add_bos=false bos_token=151643 max_abs_diff=9.196472168 rms_diff=1.153243597 max_target_abs_diff=0.111948490
== smollm2-q8 ==
compare-llamacpp-logprobs chunks=2 n_ctx=8 vocab=49152 rows=6 values=294912 add_bos=false bos_token=1 max_abs_diff=12.659570694 rms_diff=0.653486127 max_target_abs_diff=0.076397419
```

Longer-context reproducible target-model wrapper:

```sh
scripts/run-target-logprob-broad-matrix.sh --tinyllama-q8 /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q8_0.gguf --tinyllama-q4 /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --qwen25-q8 /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf --smollm2-q8 /tmp/detllm-external/SmolLM2-1.7B-Instruct-Q8_0.gguf --out /tmp/detllm-logprob-broad-matrix-20260710 --threads 8 --ctx-size 16 --batch-size 16 --chunks 3
```

This broader check uses a repeated validation prompt, `ctx-size=16`,
`chunks=3`, and the wrapper's default `--max-target-abs-diff 0.3`. The shorter
`0.2` threshold was too tight for SmolLM2 in this longer setup:
`max_target_abs_diff=0.250263214`.

Observed longer-context matrix output:

```text
== tinyllama-q8-c16-k3 ==
compare-llamacpp-logprobs chunks=3 n_ctx=16 vocab=32000 rows=21 values=672000 add_bos=true bos_token=1 max_abs_diff=13.301235199 rms_diff=1.700152315 max_target_abs_diff=0.091041565
== tinyllama-q4-c16-k3 ==
compare-llamacpp-logprobs chunks=3 n_ctx=16 vocab=32000 rows=21 values=672000 add_bos=true bos_token=1 max_abs_diff=13.530838013 rms_diff=1.578967313 max_target_abs_diff=0.152609825
== qwen25-q8-c16-k3 ==
compare-llamacpp-logprobs chunks=3 n_ctx=16 vocab=151936 rows=21 values=3190656 add_bos=false bos_token=151643 max_abs_diff=15.926328659 rms_diff=2.879756257 max_target_abs_diff=0.186036110
== smollm2-q8-c16-k3 ==
compare-llamacpp-logprobs chunks=3 n_ctx=16 vocab=49152 rows=21 values=1032192 add_bos=false bos_token=1 max_abs_diff=20.969736099 rms_diff=2.214630498 max_target_abs_diff=0.250263214
```

enwik8 first-1MB llama.cpp reference PPL smoke:

```sh
scripts/run-target-ppl-reference-matrix.sh --input /tmp/enwik8 --tinyllama-q8 /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q8_0.gguf --tinyllama-q4 /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --qwen25-q8 /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf --smollm2-q8 /tmp/detllm-external/SmolLM2-1.7B-Instruct-Q8_0.gguf --out /tmp/detllm-ppl-reference-matrix-20260710 --limit-bytes 1048576 --threads 8 --ctx-size 128 --batch-size 128 --chunks 4
```

This records external reference model-quality evidence over the same first-1MB
source used by `bench-file` preflights. It is not a detllm compression-rate
measurement. The llama.cpp build reports `build: 4847 (88b97a47)`. TinyLlama
SPM tokenization produced PPL estimates, while Qwen2.5 and SmolLM2 BPE
tokenization hit llama.cpp `invalid token = -1` on the raw enwik8 prefix and
therefore did not produce a final PPL line.

Observed PPL reference output:

```text
ppl-reference model=tinyllama-q8 status=ok ctx_size=128 chunks=4 limit_bytes=1048576 PPL = 3.9869 +/- 0.70623
ppl-reference model=tinyllama-q4 status=ok ctx_size=128 chunks=4 limit_bytes=1048576 PPL = 3.9348 +/- 0.68780
ppl-reference model=qwen25-q8 status=unavailable ctx_size=128 chunks=4 limit_bytes=1048576 reason=no-final-ppl
ppl-reference model=smollm2-q8 status=unavailable ctx_size=128 chunks=4 limit_bytes=1048576 reason=no-final-ppl
```

TinyLlama Q8_0 llama.cpp reference command:

```sh
llama-perplexity -m /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q8_0.gguf -p "Hello world from detllm validation. Hello world from detllm validation. Hello world from detllm validation. Hello world from detllm validation." --save-all-logits /tmp/llama-tiny-ppl-c8.logits --chunks 2 --threads 8 --ctx-size 8 --batch-size 8 --no-mmap --log-disable
cargo run --release -p xtask -- compare-llamacpp-logprobs --model /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q8_0.gguf --reference /tmp/llama-tiny-ppl-c8.logits --threads 8 --max-target-abs-diff 0.2
```

Observed output:

```text
compare-llamacpp-logprobs chunks=2 n_ctx=8 vocab=32000 rows=6 values=192000 add_bos=true bos_token=1 max_abs_diff=10.154125214 rms_diff=0.902546679 max_target_abs_diff=0.104429245
```

Qwen2.5 Q8_0 llama.cpp reference command:

```sh
llama-perplexity -m /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf -p "Hello world from detllm validation. Hello world from detllm validation. Hello world from detllm validation. Hello world from detllm validation." --save-all-logits /tmp/llama-qwen-ppl-c8.logits --chunks 2 --threads 8 --ctx-size 8 --batch-size 8 --no-mmap --log-disable
cargo run --release -p xtask -- compare-llamacpp-logprobs --model /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf --reference /tmp/llama-qwen-ppl-c8.logits --threads 8 --max-target-abs-diff 0.2
```

Observed output:

```text
compare-llamacpp-logprobs chunks=2 n_ctx=8 vocab=151936 rows=6 values=911616 add_bos=false bos_token=151643 max_abs_diff=9.236663818 rms_diff=1.138672226 max_target_abs_diff=0.084975481
```

Qwen2.5 Q8_0 longer 8-token text raw-logits reference command:

```sh
cargo run --release -p det-cli -- tokenize -m /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf -p "Hello world from detllm validation."
cargo run --release -p det-cli -- logits -m /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf --tokens 9707,1879,504,3392,654,76,10519,13 --dump /tmp/detllm-qwen-hello-validation-8.rawlogits.bin --hash --threads 8
/tmp/reference_logits_llamacpp --model /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf --tokens 9707,1879,504,3392,654,76,10519,13 --out /tmp/llamacpp-qwen-hello-validation-8.rawlogits.bin --threads 8 --ctx-size 16 --batch-size 16 --expected-vocab 151936 --expected-rows 8 --quiet
cargo run -p xtask -- compare-logits --actual /tmp/detllm-qwen-hello-validation-8.rawlogits.bin --reference /tmp/llamacpp-qwen-hello-validation-8.rawlogits.bin --row-size 151936 --rows 8 --min-cosine 0.999
```

Observed output:

```text
tokens("Hello world from detllm validation.") = 9707,1879,504,3392,654,76,10519,13
398b5cc327456c4c97bb515a3048a5db67777bb5266f0cdc48be3cb5c745bf41
reference_logits_llamacpp rows=8 vocab=151936 values=1215488
compare-logits values=1215488 cosine=0.999761492 max_abs_diff=0.622567177 rms_diff=0.075747794 rows=8 row_size=151936 min_row_cosine=0.999562474
```

SmolLM2 Q8_0 llama.cpp reference command:

```sh
llama-perplexity -m /tmp/detllm-external/SmolLM2-1.7B-Instruct-Q8_0.gguf -p "Hello world from detllm validation. Hello world from detllm validation. Hello world from detllm validation. Hello world from detllm validation." --save-all-logits /tmp/llama-smollm2-ppl-c8.logits --chunks 2 --threads 8 --ctx-size 8 --batch-size 8 --no-mmap --log-disable
cargo run --release -p xtask -- compare-llamacpp-logprobs --model /tmp/detllm-external/SmolLM2-1.7B-Instruct-Q8_0.gguf --reference /tmp/llama-smollm2-ppl-c8.logits --threads 8 --max-target-abs-diff 0.2
```

Observed output:

```text
compare-llamacpp-logprobs chunks=2 n_ctx=8 vocab=49152 rows=6 values=294912 add_bos=false bos_token=1 max_abs_diff=12.707466125 rms_diff=0.655142658 max_target_abs_diff=0.095531464
```

Local smoke using the same `testdata/tiny-f32.gguf` dump as both actual and
reference reported:

```text
compare-logits values=1536 cosine=1.000000000 max_abs_diff=0.000000000 rms_diff=0.000000000 rows=6 row_size=256 min_row_cosine=1.000000000
```

## Wasm Coverage

The GitHub Actions workflow also has a final `logits-hash-match` job that
downloads the hash artifacts from the three native target jobs, the two
toolchain-skew jobs, and the wasm job. It now asserts that all six artifacts
are present before comparing their labeled fixture logits hashes byte-for-byte
through:

```sh
cargo run -p xtask -- verify-logits-hashes --dir logits-hashes --expected-count 6
```

The `verify-logits-hashes` command recursively finds `logits-hashes.txt`
artifacts, requires the exact bundled fixture label set (`tiny-f32` and
`tiny-qmix`) with valid lowercase SHA-256 values, rejects missing or duplicate
labels, and checks every artifact against the first sorted reference. This is
a structural CI gate for the `detllm-design.md` §9.5 cross-platform hash-match
requirement.

Completed GitHub Actions evidence:

- Repository: `mii443/detllm`
- Commit: `ab0132e` (`Record richer bench-file evidence`)
- Run: <https://github.com/mii443/detllm/actions/runs/28955786780>
- Result: passed

The completed run passed the native matrix jobs (`x86_64-linux`,
`aarch64-macos`, `aarch64-linux`), `wasm32-wasip1`, both toolchain-skew jobs
(`stable` and `1.94.0`), `msrv`, `hygiene`, and the final
`logits-hash-match` artifact comparison. The only annotations were GitHub's
Node.js 20 deprecation notices for third-party actions; they did not affect
the result.
The hygiene job also runs:

```sh
cargo run -p xtask -- check-ci-workflow
```

That command validates the workflow structure itself: the manual
`workflow_dispatch` trigger, the three native target jobs, two toolchain-skew
hash artifacts, wasm build/execution/codec smoke, six-artifact final hash
verification, and the artifact upload names must all remain present.

The workflow also includes a `nightly-tinyllama` job for the second CI tier
described in `detllm-design.md`: it is skipped on ordinary push/PR runs and
runs only from the nightly `schedule` trigger or from `workflow_dispatch` when
`run_nightly_tinyllama=true`. That job downloads
`tinyllama-1.1b-chat-v1.0.Q8_0.gguf` from
`TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`, then runs:

```sh
cargo run -p xtask -- model-info --model "$TINYLLAMA_GGUF"
cargo run --release -p det-cli -- logits -m "$TINYLLAMA_GGUF" --tokens 1,2,3 --hash --threads 2
printf 'Hi\n' > /tmp/detllm-nightly-input.txt
cargo run --release -p det-cli -- compress -m "$TINYLLAMA_GGUF" -i /tmp/detllm-nightly-input.txt -o /tmp/detllm-nightly-output.dtlz --n-ctx 8 --threads 2
cargo run --release -p det-cli -- decompress -m "$TINYLLAMA_GGUF" -i /tmp/detllm-nightly-output.dtlz -o /tmp/detllm-nightly-restored.txt --threads 2
cmp /tmp/detllm-nightly-input.txt /tmp/detllm-nightly-restored.txt
```

`check-ci-workflow` validates that this scheduled/manual-only external GGUF
smoke remains present without adding the multi-GB download to normal push CI.

Manual `workflow_dispatch` evidence: run
<https://github.com/mii443/detllm/actions/runs/29049241175> on commit
`9907e3bd41f22287658e1113f57a331b460a96cf` completed successfully on
2026-07-09, including `nightly-tinyllama` job
<https://github.com/mii443/detllm/actions/runs/29049241175/job/86225469404>.
The job downloaded the 1,170,781,568-byte TinyLlama Q8_0 GGUF, observed model
SHA-256 `a4c9bb1dbaa372f6381a035fa5c02ef087aaa1ff1f843a56a22328114f03fc59`,
reported `tokenizer status=ok kind=sentencepiece`, `vocab status=ok`, and
`required-tensors status=ok checked=201 missing=0`, with `shape_mismatch=0`
and `unsupported_type=0`. The release-mode `logits --hash` smoke for tokens
`1,2,3` produced
`c1c6502c2705bc898a6547af2af17e58ce97382438f341abbfb9f37124fb4992`, and the
`Hi\n` compress/decompress smoke reached `cmp` successfully. Step timings were
20:49:47Z-20:53:22Z for download, 20:53:22Z-20:54:39Z for `model-info`,
20:54:39Z-20:54:43Z for release-mode `logits --hash`, and
20:54:43Z-20:54:57Z for the codec roundtrip.

The GitHub Actions `wasm` job builds `detllm` for `wasm32-wasip1`, runs
`selftest`, compares fixture `logits --hash` outputs against native execution,
and compares the quant-kernel hash against native execution. It also runs a
wasmtime DTLZ codec smoke for both bundled fixtures:

```sh
wasmtime --dir . target/wasm32-wasip1/debug/detllm.wasm compress -m testdata/tiny-f32.gguf -i testdata/tiny.tokens.txt -o wasm-codec-smoke/tiny-f32.dtlz --n-ctx 8
wasmtime --dir . target/wasm32-wasip1/debug/detllm.wasm decompress -m testdata/tiny-f32.gguf -i wasm-codec-smoke/tiny-f32.dtlz -o wasm-codec-smoke/tiny-f32.restored
cmp testdata/tiny.tokens.txt wasm-codec-smoke/tiny-f32.restored
```

The same commands run for `testdata/tiny-qmix.gguf`.

Local build-only check on 2026-07-10 after commit
`7d959ff Record latest CI success`:

```sh
cargo build -p det-cli --target wasm32-wasip1
```

This passed with the installed `wasm32-wasip1` target. `wasmtime` was not
installed in the local environment for this run, so wasm execution remains
covered by the GitHub Actions `wasm` job and the earlier local wasmtime run
recorded below.

Local run recorded with `wasmtime 46.0.1 (823d1b8f2 2026-06-24)`:

```sh
cargo build -p det-cli --target wasm32-wasip1
XDG_CACHE_HOME=/tmp/detllm-wasmtime-cache wasmtime target/wasm32-wasip1/debug/detllm.wasm selftest
XDG_CACHE_HOME=/tmp/detllm-wasmtime-cache wasmtime --dir . target/wasm32-wasip1/debug/detllm.wasm logits -m testdata/tiny-f32.gguf --tokens 0,1,2,3,0,2 --hash --chunk-size 3
XDG_CACHE_HOME=/tmp/detllm-wasmtime-cache wasmtime --dir . target/wasm32-wasip1/debug/detllm.wasm logits -m testdata/tiny-qmix.gguf --tokens 0,1,2,3,0,2 --hash --chunk-size 3
XDG_CACHE_HOME=/tmp/detllm-wasmtime-cache wasmtime target/wasm32-wasip1/debug/detllm.wasm quant-kernel-hash
```

Observed wasm logits hashes from that local run:

| check | hash |
|---|---|
| `tiny-f32` logits | `92a0280149c6b1505c84dce0d19486a2093f93b7978b579c220000d12e4ef7e7` |
| `tiny-qmix` logits | `8a34d3c4a05e9a30b90aadcdca7b6bac91655e6ab67980ccdb6726565d35f3e4` |

The same local run also compressed and decompressed `testdata/tiny.tokens.txt`
through wasmtime for both bundled fixtures and verified byte equality with
`cmp`.

The current native scalar and AVX2 SIMD quant-kernel hash is
`99832eb2ac8ddeb15731805e876a36b4013ae41c2aca0783ea02890fe9b0efba`.
Local `wasmtime` was not available for this update, so the GitHub Actions
`wasm` job remains the execution gate for the updated wasm quant-kernel value.
