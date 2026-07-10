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
cargo run -p xtask -- check-benchmark-workflow
cargo run -p xtask -- check-helper-scripts
cargo run -p xtask -- model-info --model testdata/tiny-f32.gguf
cargo run -p xtask -- model-info --model model.gguf
cargo run -p xtask -- model-info --model model-prefix.gguf --metadata-prefix
cargo run --release -p xtask -- bench-testdata --iters 100
cargo run --release -p xtask -- bench-file --model testdata/tiny-f32.gguf --input testdata/tiny.tokens.txt --n-ctx 8 --iters 2
cargo run --release -p xtask --features parallel,simd -- bench-file --model model.gguf --input enwik8 --limit-bytes 4096 --limit-tokens 512 --n-ctx 2048 --threads 8 --iters 1 --no-warmup
cargo run --release -p xtask --features parallel,simd -- bench-file --model model.gguf --input enwik8 --limit-bytes 1048576 --n-ctx 2048 --threads 8 --iters 1 --no-warmup --encode-only --show-phases --summary bench-file.summary --progress-every 100 --progress-summary bench-file.progress
scripts/run-target-full-bench.sh --model qwen25-q8.gguf --input /tmp/enwik8 --name qwen25-q8-first1m
scripts/run-target-bench-smoke.sh --input /tmp/enwik8 --tinyllama-q8 tinyllama-q8.gguf --tinyllama-q4 tinyllama-q4.gguf --qwen25-q8 qwen25-q8.gguf --smollm2-q8 smollm2-q8.gguf
scripts/run-target-logits-broad-matrix.sh --reference /tmp/reference_logits_llamacpp --tinyllama-q8 tinyllama-q8.gguf --tinyllama-q4 tinyllama-q4.gguf --qwen25-q8 qwen25-q8.gguf --smollm2-q8 smollm2-q8.gguf
scripts/run-target-hf-logits-matrix.sh --tinyllama-q8 tinyllama-q8.gguf --tinyllama-q4 tinyllama-q4.gguf --qwen25-q8 qwen25-q8.gguf --smollm2-q8 smollm2-q8.gguf
scripts/run-target-determinism-matrix.sh --tinyllama-q8 tinyllama-q8.gguf --tinyllama-q4 tinyllama-q4.gguf --qwen25-q8 qwen25-q8.gguf --smollm2-q8 smollm2-q8.gguf
scripts/run-target-codec-determinism-matrix.sh --tinyllama-q8 tinyllama-q8.gguf --tinyllama-q4 tinyllama-q4.gguf --qwen25-q8 qwen25-q8.gguf --smollm2-q8 smollm2-q8.gguf
scripts/run-target-logprob-matrix.sh --tinyllama-q8 tinyllama-q8.gguf --tinyllama-q4 tinyllama-q4.gguf --qwen25-q8 qwen25-q8.gguf --smollm2-q8 smollm2-q8.gguf
scripts/run-target-logprob-broad-matrix.sh --tinyllama-q8 tinyllama-q8.gguf --tinyllama-q4 tinyllama-q4.gguf --qwen25-q8 qwen25-q8.gguf --smollm2-q8 smollm2-q8.gguf
scripts/run-target-ppl-reference-matrix.sh --input /tmp/enwik8 --tinyllama-q8 tinyllama-q8.gguf --tinyllama-q4 tinyllama-q4.gguf --qwen25-q8 qwen25-q8.gguf --smollm2-q8 smollm2-q8.gguf
gh workflow run benchmarks.yml --repo mii443/detllm --ref main -f iters=100
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
changing the stdout result lines; progress rows include elapsed time,
throughput, remaining seconds, and estimated total seconds for the current
encode/decode phase. `--progress-summary PATH` atomically writes the latest
progress row to a file, which is useful for long runs where terminal output is
transient. `--summary PATH` also writes the final stdout summary lines to a
file via same-directory rename. `--checkpoint PATH --checkpoint-every N` can
be combined with `--output-dtlz PATH` on single-iteration runs to atomically
save the range encoder state and completed codec-symbol count during encode;
rerunning the same command resumes from that checkpoint after validating the
model SHA-256, input SHA-256, context settings, input length, and token count.
The checkpoint is removed after the final DTLZ has been written and any
round-trip verification has completed. The codec benchmark path reuses a
streaming KV cache inside each
fixed context window and only replays the configured overlap after window
rollover; repeated forward calls also reuse `ForwardWorkspace` scratch buffers
instead of allocating the large model temporaries or quantized activation
buffers per token, and codec encode computes only the selected symbol's
range-coder interval instead of materializing a full frequency/cumulative CDF
per token. Decode builds a frequency-only distribution for the range decoder
and scans it for the decoded value, avoiding the full cumulative CDF materialization
on the codec path. The hot forward path uses layout checks for already-loaded
models instead of re-scanning all weight tensors on every token and GEMV. With
the `parallel` feature, row-parallel
GEMV reuses fixed-size Rayon worker pools keyed by `--threads` instead of
spawning OS threads per matrix multiply; attention uses per-head score/prob
scratch so larger attention windows can run independent heads in parallel
while each head keeps its softmax and value accumulation order; and CDF
construction parallelizes only the independent `exp[i]` fill while keeping `Z`
and prefix sums single-threaded.
For long compression-rate runs where a separate round-trip smoke already covers
codec correctness, `bench-file --encode-only` measures payload generation
without paying for the mirrored decode pass; the default mode still verifies
round-trip byte equality. Prefix preflights can add `--estimate-full-run` to
scale the measured token throughput to the full tokenized first-1MB prefix.
`scripts/run-target-full-bench.sh` wraps the final target-model first-1MB
measurement shape and writes a stable `bench-file` summary, a DTLZ output, and
a combined progress log under `/tmp/detllm-target-bench` by default. It also
writes a `<name>.progress` file with the latest atomically replaced progress
row while the benchmark is running, plus a `<name>.checkpoint` encode
checkpoint until the run finishes successfully.
GitHub Actions also includes a `nightly-tinyllama` job that is skipped on
ordinary push/PR runs and runs only on the scheduled workflow or when
`workflow_dispatch` is started with `run_nightly_tinyllama=true`; it downloads
TinyLlama Q8_0 from Hugging Face and runs `model-info`, `logits --hash`, and a
small compress/decompress smoke. The manual run
<https://github.com/mii443/detllm/actions/runs/29049241175> passed this
external GGUF smoke on commit `9907e3b`.
`model-info` records a
lightweight GGUF
intake summary without loading all weights, including model SHA-256, parsed
config, tokenizer kind, byte coverage, vocabulary/codec compatibility, tensor
inventory, and required tensor shape/type status.

## Fixture Benchmark Snapshot

Local release benchmark on `x86_64` WSL2, AMD Ryzen 9 7950X3D, commit
`52288f1`, using:

```sh
cargo run --release -p xtask -- bench-testdata --iters 100
```

| check | throughput | note |
|---|---:|---|
| `tiny-f32` logits | 134939 tokens/s | 600 fixture tokens, hash stable |
| `tiny-qmix` logits | 126505 tokens/s | 600 fixture tokens, hash stable |
| `tiny-f32` codec | 90295 input bytes/s | 3900 bytes, round-trip verified |
| `tiny-qmix` codec | 78571 input bytes/s | 3900 bytes, round-trip verified |

These numbers are fixture-scale smoke benchmarks, not target-model compression
quality measurements.
GitHub Actions also provides a manual `benchmarks.yml` workflow for collecting
the same `bench-testdata` fixture benchmark on hosted `x86_64-linux`,
`aarch64-linux`, and `aarch64-macos` runners without adding benchmark timing
noise to normal CI.

Manual hosted snapshot, run
<https://github.com/mii443/detllm/actions/runs/29050786923> on commit
`e6136d8c9c392f84d46b53d56310399cdf15c205`, rustc `1.97.0`, `iters=100`:

| target | `tiny-f32` logits | `tiny-qmix` logits | `tiny-f32` codec | `tiny-qmix` codec |
|---|---:|---:|---:|---:|
| `x86_64-linux` | 82646 tokens/s | 80591 tokens/s | 55869 input bytes/s | 55616 input bytes/s |
| `aarch64-linux` | 119452 tokens/s | 101792 tokens/s | 75762 input bytes/s | 67332 input bytes/s |
| `aarch64-macos` | 132672 tokens/s | 105610 tokens/s | 167062 input bytes/s | 90900 input bytes/s |

Target-model prefix smoke on the same host, using
`scripts/run-target-bench-smoke.sh` with enwik8 first-1MB tokenization,
`--limit-tokens 64`, `--encode-only`, `--threads 8`, and `--n-ctx 128`:

| check | measured bytes | payload bpb | DTLZ bpb | throughput | full-token ETA |
|---|---:|---:|---:|---:|---:|
| TinyLlama Q8_0 | 169 | 6.106509 | 8.757396 | 7.093 tokens/s | 47,419 s |
| TinyLlama Q4_0 | 169 | 5.869822 | 8.520710 | 5.076 tokens/s | 66,256 s |
| Qwen2.5 Q8_0 | 190 | 0.631579 | 2.989474 | 5.280 tokens/s | 52,926 s |
| SmolLM2 Q8_0 | 162 | 0.839506 | 3.604938 | 5.480 tokens/s | 55,253 s |

This is real target-model throughput and prefix compression smoke evidence, not
the final full-token enwik8 first-1MB compression-rate result. A current
Qwen2.5 Q8_0 64-token encode-only preflight with `--estimate-full-run` reports
279,472 full tokens for the first 1MB and estimates the measured encode loop at
about 52,926 seconds, or roughly 15 hours, on this host.

Production-shape prefix round-trip on the same host, using
`scripts/run-target-full-bench.sh`, enwik8 first-1MB tokenization,
`--limit-tokens 512`, `--n-ctx 2048`, `--threads 8`, and default round-trip
mode:

| check | measured bytes | payload bpb | DTLZ bpb | round-trip throughput |
|---|---:|---:|---:|---:|
| Qwen2.5 Q8_0 | 1702 | 0.390129 | 0.653349 | 2.554 tokens/s |

The same 512-token production-shape Qwen2.5 round-trip was also run with
`--threads 8` and `--threads 16`. Both runs wrote a 139-byte DTLZ file with the
same SHA-256,
`8eb550073f2296b34c38a3192c93adb1a8c41245d08048fc812fd98d938f0ab7`,
while verifying byte round-trip internally.

Production-shape encode-only preflight with a full `n_ctx=2048` measured
prefix:

| check | tokens | measured bytes | payload bpb | DTLZ bpb | encode throughput | full-token ETA |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5 Q8_0 | 2048 | 6748 | 0.570243 | 0.636633 | 4.687 tokens/s | 59,621 s |

This extends the target-model compression-rate preflight to a complete 2048
token context window, but it is still encode-only prefix evidence rather than
the final full-token round-trip M4 acceptance measurement.

After the decode-side frequency-only CDF path, a Qwen2.5 Q8_0 64-token
round-trip smoke produced the same 71-byte DTLZ size and restored the 190-byte
token prefix through the public `decompress` command:
DTLZ SHA-256
`eab211252eb7c9af0d50ed29e0f14e5876a8de69b167505ae0807ae217a25b43`,
restored SHA-256
`b4997b129849e53a0cb6265f2561d8e57ad57003ffbcc1c7357b03918e79b03b`.

Target-model raw-logits reference smoke, using
`scripts/run-target-logits-broad-matrix.sh`, checks the same four external GGUFs
against llama.cpp on both a short low-level token stream and the tokenizer-backed
8-token validation prompt. All cases passed `--min-cosine 0.999`; the worst
minimum row cosine per model was:

| check | cases | worst min row cosine |
|---|---:|---:|
| TinyLlama Q8_0 | 2 | 0.999762988 |
| TinyLlama Q4_0 | 2 | 0.999521848 |
| Qwen2.5 Q8_0 | 2 | 0.999647692 |
| SmolLM2 Q8_0 | 2 | 0.999227139 |

`scripts/run-target-hf-logits-matrix.sh` provides the matching independent HF
Transformers raw-logits matrix shape for the same token streams and target
GGUFs. It expects a Python environment with `torch` and `transformers` plus
HF model IDs or local model directories, then writes detllm dumps, HF dumps,
and `compare-logits` logs under `/tmp/detllm-hf-logits-matrix` by default.
The HF f32-original matrix was recorded on 2026-07-10 with
`torch 2.7.1+cu126` and `transformers 5.13.0`; it is useful negative evidence
for direct HF-original-vs-quantized-GGUF comparison, because every target-model
case missed the `0.999` minimum-row threshold. Worst minimum row cosine was:

| check | cases | worst min row cosine |
|---|---:|---:|
| TinyLlama Q8_0 | 2 | 0.996110549 |
| TinyLlama Q4_0 | 2 | 0.689283705 |
| Qwen2.5 Q8_0 | 2 | 0.980463056 |
| SmolLM2 Q8_0 | 2 | 0.953664992 |

For quantized target GGUFs, the acceptance raw-logits gate is therefore the
same-GGUF llama.cpp comparison at `--min-cosine 0.999`. The HF f32-original
matrix is kept as a diagnostic and defaults to recording all rows rather than
failing at a fixed threshold.

Target-model longer-context log-probability smoke, using
`scripts/run-target-logprob-broad-matrix.sh`, checks llama.cpp
`llama-perplexity --save-all-logits` references with `ctx-size=16`,
`chunks=3`, and a repeated validation prompt. All cases passed the broad
`--max-target-abs-diff 0.3` threshold; SmolLM2 Q8_0 is the current worst case
and exceeds the shorter matrix's `0.2` threshold:

| check | rows | max target abs diff |
|---|---:|---:|
| TinyLlama Q8_0 | 21 | 0.091041565 |
| TinyLlama Q4_0 | 21 | 0.152609825 |
| Qwen2.5 Q8_0 | 21 | 0.186036110 |
| SmolLM2 Q8_0 | 21 | 0.250263214 |

Target-model llama.cpp reference PPL smoke, using
`scripts/run-target-ppl-reference-matrix.sh`, evaluates the enwik8 first-1MB
byte prefix with `ctx-size=128` and `chunks=4`. This is external model-quality
evidence, not a detllm compression-rate measurement. TinyLlama SPM models
produce PPL estimates; the BPE target models currently hit llama.cpp
`invalid token = -1` on this raw byte prefix:

| check | status | reference PPL |
|---|---|---:|
| TinyLlama Q8_0 | ok | 3.9869 +/- 0.70623 |
| TinyLlama Q4_0 | ok | 3.9348 +/- 0.68780 |
| Qwen2.5 Q8_0 | unavailable in llama.cpp PPL on raw prefix | n/a |
| SmolLM2 Q8_0 | unavailable in llama.cpp PPL on raw prefix | n/a |

Target-model determinism smoke, using
`scripts/run-target-determinism-matrix.sh`, checks the same four external GGUFs
with both the default scalar build and a `parallel,simd` build, using
`threads=1,2,7,16` and `chunk-size=1,3,8` over the tokenizer-backed 8-token
streams from the raw-logits matrix. The `chunk-size=8` case is the full-stream
prefill case for these streams. All hashes matched bit-for-bit across all 24
settings per model:

| check | logits hash |
|---|---|
| TinyLlama Q8_0 | `ded3a5204a66f58e529101511fe8d2e051fe9d71897d930ea49ec57372f3001a` |
| TinyLlama Q4_0 | `da312ede8d5c3ac7599987204c7ba954f3d86315c259c7f6c3838040cf95efb5` |
| Qwen2.5 Q8_0 | `22a98865d5bd6c45a2ae4c1a29e8b37db58a78a6c7c8caedb53a3d6baee33088` |
| SmolLM2 Q8_0 | `f9b3942c20f3a4177f8d41544a918af6cc6ec90a51c085f1f69cc73cf9f6683a` |

Target-model codec determinism smoke, using
`scripts/run-target-codec-determinism-matrix.sh`, checks DTLZ payload hashes for
the byte-escape `binary-mixed` input and the `context-spanning` input with
both the default scalar build and a `parallel,simd` build, using
`threads=1,2,7,16`. All hashes matched bit-for-bit across all 8 settings per
model/input pair, and every output also decompressed back to the original
bytes:

| check | binary-mixed DTLZ SHA-256 | context-spanning DTLZ SHA-256 |
|---|---|---|
| TinyLlama Q8_0 | `0c8551a3afa977fe51e802bc5a4810925b2707e720ed74e5cf9057f07c421092` | `d0a0b9cb671df18d6188c5bb53487a085e65869ceee07f94fe5a768a123337ee` |
| TinyLlama Q4_0 | `d2f89b70a1681bd5aaf28309e1bbc3d1f109c8ebba2c432875b7ef1b19229516` | `e79297e6e0da6e4449833057d0aaf6a6bb2b6cefe8764bc02bccb39f613f8395` |
| Qwen2.5 Q8_0 | `ea719f3444398e1e1352aee5a4ac6690ae40ce106dc1990a4a3c60a3cbe7a72c` | `7047a35e2c976cb35333e2ccc653552f94a58e77d1a884719a703d6f8b2b1fa5` |
| SmolLM2 Q8_0 | `2ac0d09372c2f16a57209a5e5bc585c8fb47913ff2bcb77588561101a61be4a4` | `960025e02a6baa87218018b877266628e37800585e5839a72d7fde6671f0d1c0` |

## Remaining Work

The implementation is not yet complete against the full design. In particular,
the following acceptance evidence is still missing:

- Further target-model reference-quality checks are still needed beyond the
  current scripted TinyLlama Q8_0/Q4_0, Qwen2.5 Q8_0, and SmolLM2 Q8_0
  llama.cpp raw-logits broad matrix and short/long log-probability matrices,
  plus the current llama.cpp PPL reference smoke. Broader perplexity-quality
  checks remain pending; direct HF-original-vs-quantized-GGUF thresholding is
  recorded as diagnostic negative evidence rather than an acceptance gate.
- Target-model enwik8 first-1MB compression-rate measurement with
  `xtask bench-file`; the bundled tiny fixture has input-scale enwik8 evidence.
- Broader benchmark results on real target-model hardware beyond the current
  bundled fixture `xtask bench-testdata` local/hosted snapshots and the current
  64-token target-model enwik8 prefix smoke matrix.
