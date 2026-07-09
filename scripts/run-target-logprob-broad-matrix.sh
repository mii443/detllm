#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: scripts/run-target-logprob-broad-matrix.sh --tinyllama-q8 PATH --tinyllama-q4 PATH --qwen25-q8 PATH --smollm2-q8 PATH [--llama-perplexity llama-perplexity] [--out DIR] [--threads N] [--ctx-size N] [--batch-size N] [--chunks N] [--max-target-abs-diff X]

Runs a longer-context target-model llama.cpp log-probability reference matrix.
Each row uses llama-perplexity --save-all-logits on a repeated validation
prompt, then compares detllm target-token log probabilities with xtask
compare-llamacpp-logprobs. The default target-token absolute-difference
threshold is 0.3 because this longer ctx=16/chunks=3 smoke is intentionally
broader than the shorter ctx=8/chunks=2 matrix.
USAGE
}

llama_perplexity="llama-perplexity"
out_dir="/tmp/detllm-logprob-broad-matrix"
threads="8"
ctx_size="16"
batch_size="16"
chunks="3"
max_target_abs_diff="0.3"
tinyllama_q8=""
tinyllama_q4=""
qwen25_q8=""
smollm2_q8=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --llama-perplexity)
      llama_perplexity="${2:?missing value for --llama-perplexity}"
      shift 2
      ;;
    --out)
      out_dir="${2:?missing value for --out}"
      shift 2
      ;;
    --threads)
      threads="${2:?missing value for --threads}"
      shift 2
      ;;
    --ctx-size)
      ctx_size="${2:?missing value for --ctx-size}"
      shift 2
      ;;
    --batch-size)
      batch_size="${2:?missing value for --batch-size}"
      shift 2
      ;;
    --chunks)
      chunks="${2:?missing value for --chunks}"
      shift 2
      ;;
    --max-target-abs-diff)
      max_target_abs_diff="${2:?missing value for --max-target-abs-diff}"
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

if [[ -z "$tinyllama_q8" || -z "$tinyllama_q4" || -z "$qwen25_q8" || -z "$smollm2_q8" ]]; then
  usage >&2
  exit 2
fi

if ! command -v "$llama_perplexity" >/dev/null 2>&1; then
  echo "missing executable: $llama_perplexity" >&2
  exit 1
fi

for path in "$tinyllama_q8" "$tinyllama_q4" "$qwen25_q8" "$smollm2_q8"; do
  if [[ ! -f "$path" ]]; then
    echo "missing file: $path" >&2
    exit 1
  fi
done

mkdir -p "$out_dir"

prompt="Hello world from detllm validation. Deterministic compression checks compare target-token log probabilities across repeated context windows. Hello world from detllm validation. Deterministic compression checks compare target-token log probabilities across repeated context windows. Hello world from detllm validation. Deterministic compression checks compare target-token log probabilities across repeated context windows. Hello world from detllm validation. Deterministic compression checks compare target-token log probabilities across repeated context windows."

run_case() {
  local name="$1"
  local model="$2"
  local reference="$out_dir/$name.llamacpp.logits"
  local compare_log="$out_dir/$name.compare.txt"

  echo "== $name =="
  "$llama_perplexity" \
    -m "$model" \
    -p "$prompt" \
    --save-all-logits "$reference" \
    --chunks "$chunks" \
    --threads "$threads" \
    --ctx-size "$ctx_size" \
    --batch-size "$batch_size" \
    --no-mmap \
    --log-disable
  cargo run --release -p xtask -- compare-llamacpp-logprobs \
    --model "$model" \
    --reference "$reference" \
    --threads "$threads" \
    --max-target-abs-diff "$max_target_abs_diff" | tee "$compare_log"
}

run_case "tinyllama-q8-c${ctx_size}-k${chunks}" "$tinyllama_q8"
run_case "tinyllama-q4-c${ctx_size}-k${chunks}" "$tinyllama_q4"
run_case "qwen25-q8-c${ctx_size}-k${chunks}" "$qwen25_q8"
run_case "smollm2-q8-c${ctx_size}-k${chunks}" "$smollm2_q8"
