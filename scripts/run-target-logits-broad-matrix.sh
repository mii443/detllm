#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: scripts/run-target-logits-broad-matrix.sh --reference /tmp/reference_logits_llamacpp --tinyllama-q8 PATH --tinyllama-q4 PATH --qwen25-q8 PATH --smollm2-q8 PATH [--bin target/release/detllm] [--out DIR] [--threads N] [--ctx-size N] [--batch-size N] [--min-cosine X]

Runs broader target-model raw-logits cosine checks against a llama.cpp
reference binary. Each model is checked on a short low-level token stream and
on the tokenizer-backed 8-token validation prompt, with per-case detllm dumps,
reference dumps, and compare-logits logs written under --out.
USAGE
}

bin="target/release/detllm"
reference=""
out_dir="/tmp/detllm-logits-broad-matrix"
threads="8"
ctx_size="16"
batch_size="16"
min_cosine="0.999"
tinyllama_q8=""
tinyllama_q4=""
qwen25_q8=""
smollm2_q8=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bin)
      bin="${2:?missing value for --bin}"
      shift 2
      ;;
    --reference)
      reference="${2:?missing value for --reference}"
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
    --min-cosine)
      min_cosine="${2:?missing value for --min-cosine}"
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

if [[ -z "$reference" || -z "$tinyllama_q8" || -z "$tinyllama_q4" || -z "$qwen25_q8" || -z "$smollm2_q8" ]]; then
  usage >&2
  exit 2
fi

for path in "$bin" "$reference" "$tinyllama_q8" "$tinyllama_q4" "$qwen25_q8" "$smollm2_q8"; do
  if [[ ! -f "$path" ]]; then
    echo "missing file: $path" >&2
    exit 1
  fi
done

mkdir -p "$out_dir"

run_case() {
  local model_name="$1"
  local model="$2"
  local vocab="$3"
  local case_name="$4"
  local rows="$5"
  local tokens="$6"
  local stem="$out_dir/$model_name-$case_name"
  local actual="$stem.detllm.rawlogits.bin"
  local expected="$stem.llamacpp.rawlogits.bin"
  local compare_log="$stem.compare.txt"

  echo "== $model_name $case_name =="
  "$bin" logits -m "$model" --tokens "$tokens" --dump "$actual" --hash --threads "$threads"
  "$reference" \
    --model "$model" \
    --tokens "$tokens" \
    --out "$expected" \
    --threads "$threads" \
    --ctx-size "$ctx_size" \
    --batch-size "$batch_size" \
    --expected-vocab "$vocab" \
    --expected-rows "$rows" \
    --sequential \
    --quiet
  cargo run --release -p xtask -- compare-logits \
    --actual "$actual" \
    --reference "$expected" \
    --row-size "$vocab" \
    --rows "$rows" \
    --min-cosine "$min_cosine" \
    --worst-rows "$rows" \
    --top-diffs 5 | tee "$compare_log"
}

run_case "tinyllama-q8" "$tinyllama_q8" 32000 "ids-1-2-3" 3 "1,2,3"
run_case "tinyllama-q8" "$tinyllama_q8" 32000 "hello-validation-8" 8 "10994,3186,515,1439,645,112,8845,49"
run_case "tinyllama-q4" "$tinyllama_q4" 32000 "ids-1-2-3" 3 "1,2,3"
run_case "tinyllama-q4" "$tinyllama_q4" 32000 "hello-validation-8" 8 "10994,3186,515,1439,645,112,8845,49"
run_case "qwen25-q8" "$qwen25_q8" 151936 "special-hello-special" 3 "151643,9707,151645"
run_case "qwen25-q8" "$qwen25_q8" 151936 "hello-validation-8" 8 "9707,1879,504,3392,654,76,10519,13"
run_case "smollm2-q8" "$smollm2_q8" 49152 "ids-1-2-3" 3 "1,2,3"
run_case "smollm2-q8" "$smollm2_q8" 49152 "hello-validation-8" 8 "19556,905,429,964,764,93,13132,30"
