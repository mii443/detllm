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
external HF/llama.cpp cosine-similarity sanity checks.
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
```

`model-info` is the lightweight first step for TinyLlama / SmolLM2 / Qwen2.5
GGUF validation. It parses the file, reports the model SHA-256, selected GGUF
metadata, tokenizer kind, deterministic `LlamaConfig` interpretation, tensor
type inventory, tokenizer/model/codec vocabulary compatibility, and the
required tensor shape/type status. It does not instantiate `F32Llama`, so it
can be run before a full logits or compression pass on larger external GGUFs.

Observed smoke output on the bundled F32 fixture:

```text
model-info path=testdata/tiny-f32.gguf bytes=13520 sha256=ce2aa01900a63585a409ef995a2827dcac81e1678e38a1ab0733302ba82ce79b gguf_version=3 metadata=12 tensors=12 data_offset=4800
model-info metadata key=general.architecture string=llama
model-info metadata key=llama.vocab_size u32=256
model-info metadata key=tokenizer.ggml.tokens array<string>[256]
model-info tokenizer status=ok kind=byte_fallback
model-info config status=ok block_count=1 embedding_length=4 feed_forward_length=6 head_count=2 head_count_kv=1 context_length=16 rope_dimension_count=2 rope_pairing=Adjacent rope_freq_base=10000.0 rms_epsilon=1e-5 attention_scale=0.70710677
model-info tensor-inventory total=12 encoded_bytes=8720 encoded_len_errors=0 F32=12
model-info vocab status=ok tokenizer=256 model=256 codec_max_symbols=262144
model-info required-tensors status=ok checked=12 missing=0 shape_mismatch=0 unsupported_type=0 tied_output=false
```

For external validation, record this output next to the logits hash, cosine
comparison, and `bench-file` output. The required tensor status is evidence
that the target model has the tensor names, dimensions, and tensor types that
the deterministic inference path will attempt to load.

## File Codec Bench Harness

Command:

```sh
cargo run --release -p xtask -- bench-file --model testdata/tiny-f32.gguf --input testdata/tiny.tokens.txt --n-ctx 8 --iters 1
cargo run --release -p xtask -- bench-file --model model.gguf --input enwik8 --limit-bytes 1048576 --n-ctx 2048 --iters 1
```

Observed smoke output on the bundled token text fixture:

| model | model SHA-256 | input SHA-256 | measured input bytes | tokens | payload bytes | DTLZ bytes | payload bpb | DTLZ bpb | ratio |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `testdata/tiny-f32.gguf` | `ce2aa01900a63585a409ef995a2827dcac81e1678e38a1ab0733302ba82ce79b` | `bfdf7888835d22d01ce148aa49e1e766f11e3fbe8631f08215e1c9173270dbd8` | 12 | 12 | 19 | 75 | 12.666667 | 50.000000 | 6.250000 |
| `testdata/tiny-qmix.gguf` | `4adbef1f9806fb17050d4520135bf8c8b4308840637b2e27589887f7fc03338f` | `bfdf7888835d22d01ce148aa49e1e766f11e3fbe8631f08215e1c9173270dbd8` | 12 | 12 | 19 | 75 | 12.666667 | 50.000000 | 6.250000 |

`bench-file` tokenizes the input, encodes the token stream, decodes it, and
detokenizes back to bytes on every iteration. It reports payload size and DTLZ
size, including the 56-byte file header. It also reports model and measured
input SHA-256 values, source and measured input byte counts, one-iteration and
total token counts, payload-only bpb, DTLZ bpb, compression ratio, elapsed
time, bytes/s, and tokens/s. `--limit-bytes N` truncates the input to at most
the first `N` bytes before tokenization, so the enwik8 first-1MB measurement
can use `--limit-bytes 1048576` without creating a separate file. This is the
harness to use for the enwik8 first-1MB measurement once the dataset and a
real target model are available; the bundled fixtures are only smoke inputs.
The harness applies the same tokenizer/model vocabulary equality check and
`2^18` codec vocabulary bound as the CLI compression path before accepting a
model for measurement.
The unit test `xtask_codec_helpers_reject_invalid_windows` also covers the
lower-level bench codec helpers directly: their encode/decode paths reject
zero `n_ctx`, `overlap >= n_ctx`, and `n_ctx` larger than the model context
before any benchmark token stream is processed.

## Logits Cosine Compare Harness

Command:

```sh
cargo run -p det-cli -- tokenize -m model.gguf -p "prompt text"
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
transformers or llama.cpp cosine-similarity sanity check required by
`detllm-design.md` once an external reference dump is available.

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
