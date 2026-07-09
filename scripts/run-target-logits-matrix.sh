#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: scripts/run-target-logits-matrix.sh --reference /tmp/reference_logits_llamacpp --tinyllama-q8 PATH --tinyllama-q4 PATH --qwen25-q8 PATH --smollm2-q8 PATH [--bin target/release/detllm] [--out DIR] [--threads N]

Runs the tokenizer-backed 8-token raw-logits cosine matrix against a llama.cpp
reference binary. Each row writes detllm and reference logits dumps, then checks
the full distribution with xtask compare-logits --min-cosine 0.999.
USAGE
}

bin="target/release/detllm"
reference=""
out_dir="/tmp/detllm-logits-matrix"
threads="8"
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
  local name="$1"
  local model="$2"
  local vocab="$3"
  local tokens="$4"
  local actual="$out_dir/$name.detllm.rawlogits.bin"
  local expected="$out_dir/$name.llamacpp.rawlogits.bin"

  echo "== $name =="
  "$bin" logits -m "$model" --tokens "$tokens" --dump "$actual" --hash --threads "$threads"
  "$reference" --model "$model" --tokens "$tokens" --out "$expected" --threads "$threads" --ctx-size 16 --batch-size 16 --expected-vocab "$vocab" --expected-rows 8 --sequential --quiet
  cargo run --release -p xtask -- compare-logits --actual "$actual" --reference "$expected" --row-size "$vocab" --rows 8 --min-cosine 0.999 --worst-rows 8 --top-diffs 5
}

run_case "tinyllama-q8" "$tinyllama_q8" 32000 "10994,3186,515,1439,645,112,8845,49"
run_case "tinyllama-q4" "$tinyllama_q4" 32000 "10994,3186,515,1439,645,112,8845,49"
run_case "qwen25-q8" "$qwen25_q8" 151936 "9707,1879,504,3392,654,76,10519,13"
run_case "smollm2-q8" "$smollm2_q8" 49152 "19556,905,429,964,764,93,13132,30"
