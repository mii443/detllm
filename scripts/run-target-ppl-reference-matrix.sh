#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: scripts/run-target-ppl-reference-matrix.sh --input /tmp/enwik8 --tinyllama-q8 PATH --tinyllama-q4 PATH --qwen25-q8 PATH --smollm2-q8 PATH [--llama-perplexity llama-perplexity] [--out DIR] [--limit-bytes N] [--threads N] [--ctx-size N] [--batch-size N] [--chunks N]

Runs a llama.cpp reference perplexity smoke over an input byte prefix. This is
not a detllm compression-rate measurement; it records an external model-quality
reference for the same enwik8 first-1MB source used by bench-file preflights.
If llama.cpp cannot evaluate a tokenizer/input pair, the per-model summary is
marked unavailable instead of being silently treated as a PPL result.
USAGE
}

input=""
llama_perplexity="llama-perplexity"
out_dir="/tmp/detllm-ppl-reference-matrix"
limit_bytes="1048576"
threads="8"
ctx_size="128"
batch_size="128"
chunks="4"
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
    --llama-perplexity)
      llama_perplexity="${2:?missing value for --llama-perplexity}"
      shift 2
      ;;
    --out)
      out_dir="${2:?missing value for --out}"
      shift 2
      ;;
    --limit-bytes)
      limit_bytes="${2:?missing value for --limit-bytes}"
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

if ! command -v "$llama_perplexity" >/dev/null 2>&1; then
  echo "missing executable: $llama_perplexity" >&2
  exit 1
fi

for path in "$input" "$tinyllama_q8" "$tinyllama_q4" "$qwen25_q8" "$smollm2_q8"; do
  if [[ ! -f "$path" ]]; then
    echo "missing file: $path" >&2
    exit 1
  fi
done

mkdir -p "$out_dir"
prefix="$out_dir/input-first-${limit_bytes}.bin"
dd if="$input" of="$prefix" bs="$limit_bytes" count=1 status=none

run_case() {
  local name="$1"
  local model="$2"
  local log="$out_dir/$name-c${ctx_size}-k${chunks}.ppl.log"

  echo "== $name =="
  "$llama_perplexity" \
    -m "$model" \
    -f "$prefix" \
    --ctx-size "$ctx_size" \
    --chunks "$chunks" \
    --threads "$threads" \
    --batch-size "$batch_size" \
    --no-mmap \
    --no-perf 2>&1 | tee "$log"

  local ppl_line
  ppl_line="$(grep -F "Final estimate: PPL =" "$log" | tail -n 1 || true)"
  if [[ -n "$ppl_line" ]]; then
    echo "ppl-reference model=$name status=ok ctx_size=$ctx_size chunks=$chunks limit_bytes=$limit_bytes ${ppl_line#Final estimate: }"
  else
    echo "ppl-reference model=$name status=unavailable ctx_size=$ctx_size chunks=$chunks limit_bytes=$limit_bytes reason=no-final-ppl"
  fi
}

run_case "tinyllama-q8" "$tinyllama_q8"
run_case "tinyllama-q4" "$tinyllama_q4"
run_case "qwen25-q8" "$qwen25_q8"
run_case "smollm2-q8" "$smollm2_q8"
