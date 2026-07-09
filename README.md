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
buffers per token, and the codec CDF
path reuses frequency/cumulative buffers across tokens while decode skips the
full validation scan for CDFs built by the codec path. The hot forward path
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

## Remaining Work

The implementation is not yet complete against the full design. In particular,
the following acceptance evidence is still missing:

- Broader target-model reference-quality checks are still needed beyond the
  current scripted TinyLlama Q8_0/Q4_0, Qwen2.5 Q8_0, and SmolLM2 Q8_0
  tokenizer-backed 8-token raw-logits matrix, logits/log-probability smoke
  evidence, and the scripted target-model
  empty/multilingual/binary/context-spanning round-trip matrix.
- Target-model enwik8 first-1MB compression-rate measurement with
  `xtask bench-file`; the bundled tiny fixture has input-scale enwik8 evidence.
- Broader benchmark results on real target hardware beyond the current bundled
  fixture `xtask bench-testdata` snapshot.
