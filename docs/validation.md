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
Those model-loading paths also reject tokenizers without a complete byte
fallback mapping for all `0x00..0xff` values. The unit test
`loaded_model_rejects_tokenizer_without_complete_byte_fallback` covers the v1
rule that models unable to losslessly represent arbitrary input bytes are not
accepted for compression or decompression.
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
7c70ec844f5cba8f140f6e8439c4ce2bf40caa2bb72d70f8a93ce11a2cfa810e
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
The CLI startup runtime canary also hashes a fixed set of Q8_0/Q4_0 block-dot
outputs before executing normal commands, so a broken selected quantized dot
backend is caught by `selftest` and by ordinary CLI entry points, not only by
the separate `quant-kernel-hash` diagnostic command.
The `shared_q8a_path_matches_standalone_quantized_gemv` test fixes the
`detllm-design.md` §5.2 quantization timing rule: one Q8A activation buffer is
created for mixed F32/quantized projection groups when any matrix needs it,
quantized GEMV requires that shared buffer, and the shared-buffer results match
standalone quantized GEMV bit-for-bit.

The AVX2 SIMD kernel path is also executed directly in CI with:

```sh
RUSTFLAGS="-C target-feature=+avx2" cargo test -p det-quant --features simd simd_blocks_match_scalar_bits
```

Local run on `x86_64` passed this test, which compares Q8_0/Q4_0 SIMD block
dots against the scalar implementation by exact `f32::to_bits()` equality.

## Determinism Hygiene

Command:

```sh
cargo run -p xtask -- check-determinism
```

The check scans implementation and CI files for `detllm-design.md` banned
constructs such as platform transcendental calls, `mul_add`, randomized
`HashMap`/`HashSet` usage, wasm `relaxed-simd`, and obvious parallel reduction
patterns. It intentionally excludes prose docs and the design file itself to
avoid flagging normative descriptions. The GitHub Actions `hygiene` job runs
this check after stale-testdata validation.
For `Cargo.toml` files, the same check also enforces dependency hygiene: path
dependencies are accepted, while external dependencies must use exact
`=x.y.z` versions. This keeps future third-party additions aligned with the
`detllm-design.md` requirement that numerically relevant dependencies not float
across builds.

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
unsupported flags, zero `n_ctx`, and `overlap >= n_ctx` are rejected before a
header is accepted or written by the CLI.
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
- Incompatible intake probe:
  `tinyllama-1.1b-chat-v1.0.Q4_0.gguf`

The Q4_0 file was useful for parser coverage but is not a v1 inference target:
it contains one `Q6_K` tensor for `output.weight`. `det-gguf` now knows the
standard GGML K-quant tensor block sizes so `model-info` can parse and report
this accurately, while `det-model` still rejects the tensor for inference
because `detllm-design.md` v1 only supports `F32`, `F16` dense loading,
`Q8_0`, and `Q4_0` inference tensors.

Observed Q4_0 intake result:

```text
model-info path=/tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q4_0.gguf bytes=637699456 sha256=da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556 metadata_prefix=false gguf_version=3 metadata=23 tensors=201 data_offset=1709440
model-info tensor-inventory total=201 encoded_bytes=635990016 encoded_len_errors=0 F32=45 Q4_0=155 Q6_K=1
model-info tensor-issue name=output.weight issue=unsupported_type type=Q6_K
model-info required-tensors status=error checked=201 missing=0 shape_mismatch=0 unsupported_type=1 tied_output=false
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
model-info tokenizer status=error error=IncompleteByteFallback
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
model-info tokenizer status=error error=IncompleteByteFallback
model-info byte-coverage tokens=49152 single_byte=235 emittable_single_byte=235 missing=21 missing_emittable=21 missing_first=04,06,13,14,16,1d,c0,c1,f1,f2,f5,f6,f7,f8,f9,fa,... missing_emittable_first=04,06,13,14,16,1d,c0,c1,f1,f2,f5,f6,f7,f8,f9,fa,...
model-info tensor-inventory total=218 encoded_bytes=1818632192 encoded_len_errors=0 F32=49 Q8_0=169
model-info required-tensors status=ok checked=218 missing=0 shape_mismatch=0 unsupported_type=0 tied_output=true
```

Observed HuggingFaceTB Q4_K_M prefix result:

```text
model-info path=/tmp/smollm2-hftb-q4-prefix.gguf bytes=4194304 sha256=278ab31551e6bef87bdbdfdb6d283c7515e5059016f19dee4cc4c26d2d4ed8ae metadata_prefix=true gguf_version=3 metadata=34 tensors=218 data_offset=1782464
model-info metadata key=general.name string=Smollm2 1.7B 8k Mix7 Ep2 v2
model-info tokenizer status=error error=IncompleteByteFallback
model-info byte-coverage tokens=49152 single_byte=235 emittable_single_byte=235 missing=21 missing_emittable=21 missing_first=04,06,13,14,16,1d,c0,c1,f1,f2,f5,f6,f7,f8,f9,fa,... missing_emittable_first=04,06,13,14,16,1d,c0,c1,f1,f2,f5,f6,f7,f8,f9,fa,...
model-info tensor-inventory total=218 encoded_bytes=1053827072 encoded_len_errors=0 F32=49 Q4_K=144 Q6_K=25
model-info required-tensors status=error checked=218 missing=0 shape_mismatch=0 unsupported_type=169 tied_output=true
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

Tokenizer-backed CLI paths correctly reject this GGUF for v1 codec use:

```sh
cargo run --release -p det-cli -- tokenize -m /tmp/detllm-external/SmolLM2-1.7B-Instruct-Q8_0.gguf -p "Hello"
```

Observed output:

```text
detllm: tokenizer error: IncompleteByteFallback
```

This is real SmolLM2 GGUF evidence for model config parsing, required tensor
compatibility on Q8_0, single-token forward, chunk-size-invariant logits
hashing on a three-token stream, and a llama.cpp raw-logits cosine check. It
also records the blocking codec issue: the tested full GGUF and the two
metadata-prefix-screened public candidates expose only 235 of the 256 byte
values as single-byte BPE seed tokens, so they cannot satisfy
`detllm-design.md` §7's arbitrary-byte losslessness requirement. Full SmolLM2
codec validation needs a compatible GGUF/tokenizer source or a deterministic
tokenizer strategy that can represent all byte values without changing the
model vocabulary.

## File Codec Bench Harness

Command:

```sh
cargo run --release -p xtask -- bench-file --model testdata/tiny-f32.gguf --input testdata/tiny.tokens.txt --n-ctx 8 --iters 1
cargo run --release -p xtask --features parallel,simd -- bench-file --model model.gguf --input enwik8 --limit-bytes 4096 --limit-tokens 512 --n-ctx 2048 --threads 8 --iters 1 --no-warmup
cargo run --release -p xtask --features parallel,simd -- bench-file --model model.gguf --input enwik8 --limit-bytes 1048576 --n-ctx 2048 --threads 8 --iters 1 --no-warmup
```

Build `xtask` with `--features parallel,simd` for target-model benchmark
commands. The `parallel` feature forwards to `det-model/parallel`, so
`--threads N` engages deterministic row-parallel GEMV; `simd` forwards to the
quantized kernel feature.

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
cargo run --release -p xtask --features parallel,simd -- bench-file --model /tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf --input /tmp/enwik8 --limit-bytes 1048576 --limit-tokens 16 --n-ctx 64 --threads 8 --iters 1 --no-warmup
```

```text
bench-file model=/tmp/detllm-external/qwen2.5-1.5b-instruct-q8_0.gguf input=/tmp/enwik8 limit_bytes=1048576 limit_tokens=16 iters=1 warmup=false threads=8 n_ctx=64 overlap=16 model_sha256=d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8 input_sha256=4fe5a21798e43c8258edcf9f3a98fac2df77613b4d2add15a2a3082eedc7b0b2
bench-file: source_input_bytes=100000000 measured_input_bytes=53 total_input_bytes=53 tokens=16 total_tokens=16 payload_bytes=14 dtlz_bytes=70 payload_bits_per_byte=2.113208 dtlz_bits_per_byte=10.566038 compression_ratio=1.320755 elapsed_ms=46230.631 input_bytes_per_s=1.146 tokens_per_s=0.346
```

This is input-scale and round-trip evidence for the `bench-file`
implementation on the canonical enwik8 byte stream, not a meaningful language
model compression-quality result. The tiny fixture has byte tokens and a tiny
context, so it is expected to produce near-raw 8 bpb output.

`bench-file` tokenizes the input, encodes the token stream, decodes it, and
detokenizes back to bytes on every measured iteration. It reports payload size
and DTLZ size, including the 56-byte file header. It also reports model and
measured input SHA-256 values, source and measured input byte counts,
one-iteration and total token counts, payload-only bpb, DTLZ bpb, compression
ratio, elapsed time, bytes/s, tokens/s, whether a pre-measurement warmup
round-trip was run, and the thread override used for model kernels.
`--limit-bytes N` truncates the input to at most the first `N` bytes before
tokenization, so the enwik8 first-1MB measurement can use
`--limit-bytes 1048576` without creating a separate file. `--limit-tokens N`
then truncates the tokenized stream and detokenizes that prefix back to bytes
before measurement; this gives a reproducible target-model prefix smoke path
for long runs while keeping the reported byte counts and SHA-256 tied to the
actual bytes round-tripped. Tokenization still happens before token truncation.
The ByteBPE path uses a priority-queue merge implementation, so 1MB byte caps
are usable for Qwen2.5 prefix preflights; use smaller `--limit-bytes` values
only when an even faster smoke is needed. Omit `--limit-tokens` for the final
first-1MB acceptance measurement. `--threads N` fixes the model parallelism for
reproducible benchmark notes, and `--no-warmup` skips the extra
pre-measurement round-trip for long target-model measurements; the measured
iteration still verifies encode/decode byte round-trip. This is the harness to
use for target-model enwik8 first-1MB measurements; the bundled fixtures
remain smoke and input-scale checks.
The harness applies the same tokenizer/model vocabulary equality check and
`2^18` codec vocabulary bound as the CLI compression path before accepting a
model for measurement.
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
rustc 1.95.0 (59807616e 2026-04-14)
```

Observed output:

```text
bench-testdata iters=100
logits tiny-f32: hash=92a0280149c6b1505c84dce0d19486a2093f93b7978b579c220000d12e4ef7e7 tokens=600 elapsed_ms=5.654 tokens_per_s=106111.003
logits tiny-qmix: hash=8a34d3c4a05e9a30b90aadcdca7b6bac91655e6ab67980ccdb6726565d35f3e4 tokens=600 elapsed_ms=6.065 tokens_per_s=98925.928
codec tiny-f32: input_bytes=3900 payload_bytes=4600 elapsed_ms=175.157 input_bytes_per_s=22265.754
codec tiny-qmix: input_bytes=3900 payload_bytes=4600 elapsed_ms=193.805 input_bytes_per_s=20123.346
```

`bench-testdata` verifies that the fixture logits hash does not change during
the measured loop, and each codec benchmark decodes the measured payload and
checks byte equality. This is an equivalent harness result for the bundled
fixtures only; target-model and broader hardware benchmark results remain
separate acceptance evidence.

## Logits Cosine Compare Harness

Command:

```sh
cargo run -p det-cli -- tokenize -m model.gguf -p "prompt text"
scripts/reference_logits_transformers.py --model-id HF_MODEL_ID --tokens TOKENS --out reference.logits.bin --expected-rows TOKENS --expected-vocab VOCAB
cargo run -p xtask -- compare-logits --actual detllm.logits.bin --reference reference.logits.bin --row-size VOCAB --rows TOKENS --min-cosine 0.999
```

`detllm tokenize` emits the comma-separated token IDs produced by the same
GGUF tokenizer path used by `logits -p`, after checking that tokenizer and
model vocabulary lengths match. These IDs are the prompt-shape evidence to
record next to an external HF transformers or llama.cpp reference dump.
`compare-logits` reads two little-endian f32 logits dumps, rejects malformed
lengths and non-finite values, then reports global cosine similarity, maximum
absolute difference, and RMS difference. With `--row-size`, it also reports
the minimum row cosine across token positions and applies `--min-cosine` to
both the global cosine and minimum row cosine. With `--rows`, it rejects dumps
whose row count does not match the expected number of token positions, which
prevents comparing different prompt lengths or a single final-token dump
against a full-position detllm dump. This is the harness for the HF
transformers raw-logits cosine-similarity sanity check required by
`detllm-design.md` once an external reference dump is available.

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
explicit token IDs only, marks every token position for logits output, and can
enforce row count and vocabulary size before writing the file.

Example TinyLlama command shape:

```sh
cargo run --release -p det-cli -- logits -m /tmp/detllm-external/tinyllama-1.1b-chat-v1.0.Q8_0.gguf --tokens 1,2,3 --dump detllm.logits.bin --hash --threads 8
scripts/reference_logits_transformers.py --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tokens 1,2,3 --out reference.logits.bin --expected-rows 3 --expected-vocab 32000 --threads 1 --dtype float32
cargo run -p xtask -- compare-logits --actual detllm.logits.bin --reference reference.logits.bin --row-size 32000 --rows 3 --min-cosine 0.999
```

The script's syntax and argument parser were checked locally without writing
bytecode with:

```sh
python3 -B scripts/reference_logits_transformers.py --help
```

This still does not count as the required external raw-logits cosine evidence
until it is run in an environment with compatible `torch` and `transformers`
installed and the resulting `compare-logits` output is recorded here.

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

Local run recorded with `wasmtime 46.0.1 (823d1b8f2 2026-06-24)`:

```sh
cargo build -p det-cli --target wasm32-wasip1
XDG_CACHE_HOME=/tmp/detllm-wasmtime-cache wasmtime target/wasm32-wasip1/debug/detllm.wasm selftest
XDG_CACHE_HOME=/tmp/detllm-wasmtime-cache wasmtime --dir . target/wasm32-wasip1/debug/detllm.wasm logits -m testdata/tiny-f32.gguf --tokens 0,1,2,3,0,2 --hash --chunk-size 3
XDG_CACHE_HOME=/tmp/detllm-wasmtime-cache wasmtime --dir . target/wasm32-wasip1/debug/detllm.wasm logits -m testdata/tiny-qmix.gguf --tokens 0,1,2,3,0,2 --hash --chunk-size 3
XDG_CACHE_HOME=/tmp/detllm-wasmtime-cache wasmtime target/wasm32-wasip1/debug/detllm.wasm quant-kernel-hash
```

Observed wasm hashes:

| check | hash |
|---|---|
| `tiny-f32` logits | `92a0280149c6b1505c84dce0d19486a2093f93b7978b579c220000d12e4ef7e7` |
| `tiny-qmix` logits | `8a34d3c4a05e9a30b90aadcdca7b6bac91655e6ab67980ccdb6726565d35f3e4` |
| quant kernel | `7c70ec844f5cba8f140f6e8439c4ce2bf40caa2bb72d70f8a93ce11a2cfa810e` |

The same local run also compressed and decompressed `testdata/tiny.tokens.txt`
through wasmtime for both bundled fixtures and verified byte equality with
`cmp`.
