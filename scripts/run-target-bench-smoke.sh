#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: scripts/run-target-bench-smoke.sh --input /tmp/enwik8 --tinyllama-q8 PATH --tinyllama-q4 PATH --qwen25-q8 PATH --smollm2-q8 PATH [--limit-bytes N] [--limit-tokens N] [--n-ctx N] [--threads N] [--progress-every N]

Runs reproducible target-model bench-file prefix smokes across the current
external GGUF validation set. This is intended to record real-model throughput
and prefix compression evidence while the full enwik8 first-1MB run is still
too long for routine validation.
USAGE
}

input=""
limit_bytes="1048576"
limit_tokens="16"
n_ctx="64"
threads="8"
progress_every="8"
tinyllama_q8=""
tinyllama_q4=""
qwen25_q8=""
smollm2_q8=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      input="${2:?missing value for --input}"
      shift 2
      ;;
    --limit-bytes)
      limit_bytes="${2:?missing value for --limit-bytes}"
      shift 2
      ;;
    --limit-tokens)
      limit_tokens="${2:?missing value for --limit-tokens}"
      shift 2
      ;;
    --n-ctx)
      n_ctx="${2:?missing value for --n-ctx}"
      shift 2
      ;;
    --threads)
      threads="${2:?missing value for --threads}"
      shift 2
      ;;
    --progress-every)
      progress_every="${2:?missing value for --progress-every}"
      shift 2
      ;;
    --tinyllama-q8)
      tinyllama_q8="${2:?missing value for --tinyllama-q8}"
      shift 2
      ;;
    --tinyllama-q4)
      tinyllama_q4="${2:?missing value for --tinyllama-q4}"
      shift 2
      ;;
    --qwen25-q8)
      qwen25_q8="${2:?missing value for --qwen25-q8}"
      shift 2
      ;;
    --smollm2-q8)
      smollm2_q8="${2:?missing value for --smollm2-q8}"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$input" || -z "$tinyllama_q8" || -z "$tinyllama_q4" || -z "$qwen25_q8" || -z "$smollm2_q8" ]]; then
  usage >&2
  exit 2
fi

for path in "$input" "$tinyllama_q8" "$tinyllama_q4" "$qwen25_q8" "$smollm2_q8"; do
  if [[ ! -f "$path" ]]; then
    echo "missing file: $path" >&2
    exit 1
  fi
done

run_case() {
  local name="$1"
  local model="$2"

  echo "== $name =="
  cargo run --release -p xtask --features parallel,simd -- bench-file \
    --model "$model" \
    --input "$input" \
    --limit-bytes "$limit_bytes" \
    --limit-tokens "$limit_tokens" \
    --n-ctx "$n_ctx" \
    --threads "$threads" \
    --iters 1 \
    --no-warmup \
    --encode-only \
    --show-phases \
    --progress-every "$progress_every"
}

run_case "tinyllama-q8" "$tinyllama_q8"
run_case "tinyllama-q4" "$tinyllama_q4"
run_case "qwen25-q8" "$qwen25_q8"
run_case "smollm2-q8" "$smollm2_q8"
