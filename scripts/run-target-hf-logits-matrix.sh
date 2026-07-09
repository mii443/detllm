#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: scripts/run-target-hf-logits-matrix.sh --tinyllama-q8 PATH --tinyllama-q4 PATH --qwen25-q8 PATH --smollm2-q8 PATH [--bin target/release/detllm] [--out DIR] [--python python3] [--threads N] [--torch-threads N] [--device cpu] [--dtype float32] [--min-cosine X] [--tinyllama-hf MODEL] [--qwen25-hf MODEL] [--smollm2-hf MODEL] [--trust-remote-code]

Runs target-model raw-logits cosine diagnostics against HF Transformers. HF
model arguments may be either Hugging Face model IDs or local model
directories. Each row writes detllm dumps, HF dumps, and compare-logits logs
under --out.

The default --min-cosine is 0.0 so quantized GGUF vs original HF f32 model
comparisons are recorded instead of treated as acceptance failures. Use an
explicit threshold, for example --min-cosine 0.999, only when comparing an
equivalent f32 path or intentionally checking a calibrated threshold.

This wrapper requires a Python environment with torch and transformers installed
and enough local cache/network access for the requested HF model IDs.
USAGE
}

bin="target/release/detllm"
out_dir="/tmp/detllm-hf-logits-matrix"
python_bin="python3"
threads="8"
torch_threads="1"
device="cpu"
dtype="float32"
min_cosine="0.0"
trust_remote_code=()
tinyllama_hf="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
qwen25_hf="Qwen/Qwen2.5-1.5B-Instruct"
smollm2_hf="HuggingFaceTB/SmolLM2-1.7B-Instruct"
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
    --out)
      out_dir="${2:?missing value for --out}"
      shift 2
      ;;
    --python)
      python_bin="${2:?missing value for --python}"
      shift 2
      ;;
    --threads)
      threads="${2:?missing value for --threads}"
      shift 2
      ;;
    --torch-threads)
      torch_threads="${2:?missing value for --torch-threads}"
      shift 2
      ;;
    --device)
      device="${2:?missing value for --device}"
      shift 2
      ;;
    --dtype)
      dtype="${2:?missing value for --dtype}"
      shift 2
      ;;
    --min-cosine)
      min_cosine="${2:?missing value for --min-cosine}"
      shift 2
      ;;
    --tinyllama-hf)
      tinyllama_hf="${2:?missing value for --tinyllama-hf}"
      shift 2
      ;;
    --qwen25-hf)
      qwen25_hf="${2:?missing value for --qwen25-hf}"
      shift 2
      ;;
    --smollm2-hf)
      smollm2_hf="${2:?missing value for --smollm2-hf}"
      shift 2
      ;;
    --trust-remote-code)
      trust_remote_code=(--trust-remote-code)
      shift
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

for path in "$bin" "$tinyllama_q8" "$tinyllama_q4" "$qwen25_q8" "$smollm2_q8" scripts/reference_logits_transformers.py; do
  if [[ ! -f "$path" ]]; then
    echo "missing file: $path" >&2
    exit 1
  fi
done

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "missing executable: $python_bin" >&2
  exit 1
fi

mkdir -p "$out_dir"

run_case() {
  local model_name="$1"
  local model="$2"
  local hf_model="$3"
  local vocab="$4"
  local case_name="$5"
  local rows="$6"
  local tokens="$7"
  local stem="$out_dir/$model_name-$case_name"
  local actual="$stem.detllm.rawlogits.bin"
  local expected="$stem.hf.rawlogits.bin"
  local compare_log="$stem.compare.txt"

  echo "== $model_name $case_name =="
  "$bin" logits -m "$model" --tokens "$tokens" --dump "$actual" --hash --threads "$threads"
  "$python_bin" -B scripts/reference_logits_transformers.py \
    --model-id "$hf_model" \
    --tokens "$tokens" \
    --out "$expected" \
    --device "$device" \
    --dtype "$dtype" \
    --threads "$torch_threads" \
    --expected-vocab "$vocab" \
    --expected-rows "$rows" \
    "${trust_remote_code[@]}"
  cargo run --release -p xtask -- compare-logits \
    --actual "$actual" \
    --reference "$expected" \
    --row-size "$vocab" \
    --rows "$rows" \
    --min-cosine "$min_cosine" \
    --worst-rows "$rows" \
    --top-diffs 5 | tee "$compare_log"
}

run_case "tinyllama-q8" "$tinyllama_q8" "$tinyllama_hf" 32000 "ids-1-2-3" 3 "1,2,3"
run_case "tinyllama-q8" "$tinyllama_q8" "$tinyllama_hf" 32000 "hello-validation-8" 8 "10994,3186,515,1439,645,112,8845,49"
run_case "tinyllama-q4" "$tinyllama_q4" "$tinyllama_hf" 32000 "ids-1-2-3" 3 "1,2,3"
run_case "tinyllama-q4" "$tinyllama_q4" "$tinyllama_hf" 32000 "hello-validation-8" 8 "10994,3186,515,1439,645,112,8845,49"
run_case "qwen25-q8" "$qwen25_q8" "$qwen25_hf" 151936 "special-hello-special" 3 "151643,9707,151645"
run_case "qwen25-q8" "$qwen25_q8" "$qwen25_hf" 151936 "hello-validation-8" 8 "9707,1879,504,3392,654,76,10519,13"
run_case "smollm2-q8" "$smollm2_q8" "$smollm2_hf" 49152 "ids-1-2-3" 3 "1,2,3"
run_case "smollm2-q8" "$smollm2_q8" "$smollm2_hf" 49152 "hello-validation-8" 8 "19556,905,429,964,764,93,13132,30"
