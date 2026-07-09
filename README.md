# detllm

`detllm` is a deterministic Rust LLM inference and lossless compression
prototype based on the normative design in [detllm-design.md](detllm-design.md).
The current implementation focuses on bit-identical CPU logits/CDF generation
for a Llama-style decoder model, GGUF `F32` / `Q8_0` / `Q4_0` / `Q4_K` /
`Q6_K` tensor loading, and a range-coder-backed `compress` / `decompress` CLI.

## Current Status

Implemented crates:

- `det-num`: fixed-order reductions, deterministic rounding, f16 conversion,
  vendored libm `exp`/`sin`/`cos`/`log`, SHA-256, numeric canary.
- `det-quant`: `Q8_0`, `Q4_0`, scalar `Q4_K`, `Q6_K`, in-memory `Q8A`/`Q8_K`,
  `simd` feature kernels, and deterministic quant-kernel hash coverage.
- `det-gguf`: zero-copy GGUF metadata and tensor parsing for repository
  fixtures.
- `det-token`: byte fallback, SentencePiece-style, and GPT-2-style tokenizer
  paths used by the v1 target models.
- `det-model`: deterministic Llama-style single-token forward pass, RMSNorm,
  RoPE, GQA attention, SwiGLU, F32/Q8/Q4/Q4_K/Q6_K GEMV, `parallel` feature
  row partitioning, and logits hashing.
- `det-coder`: logits-to-CDF conversion, 64-bit range coder, and DTLZ header.
- `det-cli`: `selftest`, `gguf-dump`, `sha256`, `tokenize`, `logits`,
  `compress`, and `decompress`.
- `xtask`: deterministic generation and stale-checking of repository testdata.

Repository fixtures:

| fixture | purpose | logits hash |
|---|---|---|
| `testdata/tiny-f32.gguf` | all-F32 reference path | `92a0280149c6b1505c84dce0d19486a2093f93b7978b579c220000d12e4ef7e7` |
| `testdata/tiny-qmix.gguf` | mixed `Q8_0`/`Q4_0` path | `8a34d3c4a05e9a30b90aadcdca7b6bac91655e6ab67980ccdb6726565d35f3e4` |

The shared token sequence is `testdata/tiny.tokens.txt`.

## Common Commands

```sh
cargo run -p xtask -- generate-testdata --check
cargo run -p xtask -- check-determinism
cargo run -p xtask -- check-ci-workflow
cargo run -p xtask -- model-info --model testdata/tiny-f32.gguf
cargo run -p xtask -- model-info --model model.gguf
cargo run -p xtask -- model-info --model model-prefix.gguf --metadata-prefix
cargo run --release -p xtask -- bench-testdata --iters 100
cargo run --release -p xtask -- bench-file --model testdata/tiny-f32.gguf --input testdata/tiny.tokens.txt --n-ctx 8 --iters 2
cargo run --release -p xtask --features parallel,simd -- bench-file --model model.gguf --input enwik8 --limit-bytes 4096 --limit-tokens 512 --n-ctx 2048 --threads 8 --iters 1 --no-warmup
cargo run --release -p xtask --features parallel,simd -- bench-file --model model.gguf --input enwik8 --limit-bytes 1048576 --n-ctx 2048 --threads 8 --iters 1 --no-warmup --encode-only --show-phases --progress-every 100
scripts/run-target-bench-smoke.sh --input /tmp/enwik8 --tinyllama-q8 tinyllama-q8.gguf --tinyllama-q4 tinyllama-q4.gguf --qwen25-q8 qwen25-q8.gguf --smollm2-q8 smollm2-q8.gguf
scripts/run-target-determinism-matrix.sh --tinyllama-q8 tinyllama-q8.gguf --tinyllama-q4 tinyllama-q4.gguf --qwen25-q8 qwen25-q8.gguf --smollm2-q8 smollm2-q8.gguf
scripts/run-target-codec-determinism-matrix.sh --tinyllama-q8 tinyllama-q8.gguf --tinyllama-q4 tinyllama-q4.gguf --qwen25-q8 qwen25-q8.gguf --smollm2-q8 smollm2-q8.gguf
cargo run -p det-cli -- tokenize -m model.gguf -p "prompt text"
scripts/reference_logits_transformers.py --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tokens 1,2,3 --out hf.logits.bin --expected-rows 3 --expected-vocab 32000
c++ -std=c++17 -O2 -I/usr/local/include scripts/reference_logits_llamacpp.cpp -L/usr/local/lib -Wl,-rpath,/usr/local/lib -lllama -lggml -lggml-cpu -lggml-base -o /tmp/reference_logits_llamacpp
/tmp/reference_logits_llamacpp --model model.gguf --tokens 1,2,3 --out llama.logits.bin --expected-rows 3 --expected-vocab VOCAB --quiet
cargo run -p xtask -- compare-logits --actual detllm.logits.bin --reference reference.logits.bin --row-size VOCAB --rows TOKENS --min-cosine 0.999 --worst-rows 3 --top-diffs 10
llama-perplexity -m model.gguf -p "prompt text long enough for 2*n_ctx tokens" --save-all-logits llama.logits --ctx-size 8 --chunks 2 --batch-size 8
cargo run --release -p xtask -- compare-llamacpp-logprobs --model model.gguf --reference llama.logits --max-target-abs-diff 0.2
cargo run -p det-cli -- selftest
cargo run -p det-cli -- logits -m testdata/tiny-f32.gguf --tokens "$(cat testdata/tiny.tokens.txt)" --hash --chunk-size 3
cargo run -p det-cli -- logits -m testdata/tiny-qmix.gguf --tokens "$(cat testdata/tiny.tokens.txt)" --hash --chunk-size 3
cargo test --workspace
cargo test --workspace --features parallel,simd
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --all-targets --features parallel,simd -- -D warnings
```

## Local Validation Snapshot

The current smoke validation is recorded in
[docs/validation.md](docs/validation.md). The included compression smoke uses
the tiny F32 fixture and verifies byte-for-byte round-trip on a small input; it
is not a meaningful compression-ratio benchmark. `bench-file` records model and
input SHA-256 values, measured byte/token counts, tokenized count before
`--limit-tokens`, payload and DTLZ bpb, compression ratio, throughput, optional
codec-symbol prefix limit, warmup mode, measurement mode, and thread override so real enwik8
measurements can be copied directly into the validation notes. Tokenizers that
cannot emit all 256 byte values use deterministic byte escape symbols after the
model vocabulary for codec round-trip, with the model context advanced only for
real vocabulary tokens; new DTLZ files set the `FLAG_BYTE_ESCAPES` header bit
so decoders can distinguish this CDF alphabet from legacy token-only payloads.
Long target-model measurements can use
`--progress-every N` to emit encode/decode token progress on stderr without
changing the stdout result lines. The codec benchmark path reuses a
streaming KV cache inside each
fixed context window and only replays the configured overlap after window
rollover; repeated forward calls also reuse `ForwardWorkspace` scratch buffers
instead of allocating the large model temporaries or quantized activation
buffers per token, and codec encode computes only the selected symbol's
range-coder interval instead of materializing a full frequency/cumulative CDF
per token. Decode still reuses frequency/cumulative buffers across tokens and
skips the full validation scan for CDFs built by the codec path. The hot forward path
uses layout checks for already-loaded models instead of re-scanning all weight
tensors on every token and GEMV. With the `parallel` feature, row-parallel
GEMV reuses fixed-size Rayon worker pools keyed by `--threads` instead of
spawning OS threads per matrix multiply; attention uses per-head score/prob
scratch so larger attention windows can run independent heads in parallel
while each head keeps its softmax and value accumulation order; and CDF
construction parallelizes only the independent `exp[i]` fill while keeping `Z`
and prefix sums single-threaded.
For long compression-rate runs where a separate round-trip smoke already covers
codec correctness, `bench-file --encode-only` measures payload generation
without paying for the mirrored decode pass; the default mode still verifies
round-trip byte equality.
`model-info` records a
lightweight GGUF
intake summary without loading all weights, including model SHA-256, parsed
config, tokenizer kind, byte coverage, vocabulary/codec compatibility, tensor
inventory, and required tensor shape/type status.

## Fixture Benchmark Snapshot

Local release benchmark on `x86_64` WSL2, AMD Ryzen 9 7950X3D, rustc
`1.95.0`, using:

```sh
cargo run --release -p xtask -- bench-testdata --iters 100
```

| check | throughput | note |
|---|---:|---|
| `tiny-f32` logits | 106111 tokens/s | 600 fixture tokens, hash stable |
| `tiny-qmix` logits | 98926 tokens/s | 600 fixture tokens, hash stable |
| `tiny-f32` codec | 22266 input bytes/s | 3900 bytes, round-trip verified |
| `tiny-qmix` codec | 20123 input bytes/s | 3900 bytes, round-trip verified |

These numbers are fixture-scale smoke benchmarks, not target-model compression
quality measurements.

Target-model prefix smoke on the same host, using
`scripts/run-target-bench-smoke.sh` with enwik8 first-1MB tokenization,
`--limit-tokens 16`, `--encode-only`, `--threads 8`, and `--n-ctx 64`:

| check | measured bytes | payload bpb | DTLZ bpb | throughput |
|---|---:|---:|---:|---:|
| TinyLlama Q8_0 | 47 | 5.617021 | 15.148936 | 7.377 tokens/s |
| TinyLlama Q4_0 | 47 | 5.787234 | 15.319149 | 5.285 tokens/s |
| Qwen2.5 Q8_0 | 53 | 1.962264 | 10.415094 | 5.366 tokens/s |
| SmolLM2 Q8_0 | 46 | 2.434783 | 12.173913 | 5.352 tokens/s |

This is real target-model throughput and prefix compression smoke evidence, not
the final full-token enwik8 first-1MB compression-rate result.

Target-model determinism smoke, using
`scripts/run-target-determinism-matrix.sh`, checks the same four external GGUFs
with `threads=1,2,8` and `chunk-size=1,2,8` over the tokenizer-backed 8-token
streams from the raw-logits matrix. All hashes matched bit-for-bit:

| check | logits hash |
|---|---|
| TinyLlama Q8_0 | `ded3a5204a66f58e529101511fe8d2e051fe9d71897d930ea49ec57372f3001a` |
| TinyLlama Q4_0 | `da312ede8d5c3ac7599987204c7ba954f3d86315c259c7f6c3838040cf95efb5` |
| Qwen2.5 Q8_0 | `22a98865d5bd6c45a2ae4c1a29e8b37db58a78a6c7c8caedb53a3d6baee33088` |
| SmolLM2 Q8_0 | `f9b3942c20f3a4177f8d41544a918af6cc6ec90a51c085f1f69cc73cf9f6683a` |

Target-model codec determinism smoke, using
`scripts/run-target-codec-determinism-matrix.sh`, checks DTLZ payload hashes for
the byte-escape `binary-mixed` input and the `context-spanning` input with
`threads=1,2,8`. Every output also decompresses back to the original bytes:

| check | binary-mixed DTLZ SHA-256 | context-spanning DTLZ SHA-256 |
|---|---|---|
| TinyLlama Q8_0 | `0c8551a3afa977fe51e802bc5a4810925b2707e720ed74e5cf9057f07c421092` | `d0a0b9cb671df18d6188c5bb53487a085e65869ceee07f94fe5a768a123337ee` |
| TinyLlama Q4_0 | `d2f89b70a1681bd5aaf28309e1bbc3d1f109c8ebba2c432875b7ef1b19229516` | `e79297e6e0da6e4449833057d0aaf6a6bb2b6cefe8764bc02bccb39f613f8395` |
| Qwen2.5 Q8_0 | `ea719f3444398e1e1352aee5a4ac6690ae40ce106dc1990a4a3c60a3cbe7a72c` | `7047a35e2c976cb35333e2ccc653552f94a58e77d1a884719a703d6f8b2b1fa5` |
| SmolLM2 Q8_0 | `2ac0d09372c2f16a57209a5e5bc585c8fb47913ff2bcb77588561101a61be4a4` | `960025e02a6baa87218018b877266628e37800585e5839a72d7fde6671f0d1c0` |

## Remaining Work

The implementation is not yet complete against the full design. In particular,
the following acceptance evidence is still missing:

- Broader target-model reference-quality checks are still needed beyond the
  current scripted TinyLlama Q8_0/Q4_0, Qwen2.5 Q8_0, and SmolLM2 Q8_0
  tokenizer-backed 8-token raw-logits matrix, logits/log-probability smoke
  evidence, and the scripted target-model
  empty/multilingual/binary/context-spanning round-trip and
  logits/thread/chunk-size and codec/thread determinism matrices.
- Target-model enwik8 first-1MB compression-rate measurement with
  `xtask bench-file`; the bundled tiny fixture has input-scale enwik8 evidence.
- Broader benchmark results on real target hardware beyond the current bundled
  fixture `xtask bench-testdata` snapshot and the current 16-token target-model
  enwik8 prefix smoke matrix.
