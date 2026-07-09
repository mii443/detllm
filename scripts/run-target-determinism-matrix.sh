#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: scripts/run-target-determinism-matrix.sh --tinyllama-q8 PATH --tinyllama-q4 PATH --qwen25-q8 PATH --smollm2-q8 PATH [--bin target/release/detllm]

Checks target-model logits hash invariance across deterministic chunking and
thread-count settings. The matrix uses the same tokenizer-backed 8-token
streams as run-target-logits-matrix.sh, and requires every hash for each model
to match bit-for-bit. The default matrix follows detllm-design.md section 9.2:
threads={1,2,7,16} and chunk-size={1,3,full}, where full is 8 for these
8-token validation streams.
USAGE
}

bin="target/release/detllm"
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

for path in "$bin" "$tinyllama_q8" "$tinyllama_q4" "$qwen25_q8" "$smollm2_q8"; do
  if [[ ! -f "$path" ]]; then
    echo "missing file: $path" >&2
    exit 1
  fi
done

run_case() {
  local name="$1"
  local model="$2"
  local tokens="$3"
  local baseline=""

  echo "== $name =="
  for threads in 1 2 7 16; do
    for chunk_size in 1 3 8; do
      local hash
      hash="$("$bin" logits -m "$model" --tokens "$tokens" --hash --threads "$threads" --chunk-size "$chunk_size")"
      if [[ -z "$baseline" ]]; then
        baseline="$hash"
      elif [[ "$hash" != "$baseline" ]]; then
        echo "hash mismatch for $name threads=$threads chunk-size=$chunk_size: got $hash expected $baseline" >&2
        exit 1
      fi
      echo "hash model=$name threads=$threads chunk_size=$chunk_size value=$hash"
    done
  done
  echo "ok model=$name hash=$baseline"
}

run_case "tinyllama-q8" "$tinyllama_q8" "10994,3186,515,1439,645,112,8845,49"
run_case "tinyllama-q4" "$tinyllama_q4" "10994,3186,515,1439,645,112,8845,49"
run_case "qwen25-q8" "$qwen25_q8" "9707,1879,504,3392,654,76,10519,13"
run_case "smollm2-q8" "$smollm2_q8" "19556,905,429,964,764,93,13132,30"
