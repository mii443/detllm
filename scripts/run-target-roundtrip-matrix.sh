#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: scripts/run-target-roundtrip-matrix.sh --tinyllama-q8 PATH --tinyllama-q4 PATH --qwen25-q8 PATH --smollm2-q8 PATH [--bin target/release/detllm] [--out DIR] [--threads N] [--n-ctx N]

Runs the external target-model round-trip matrix required by detllm-design.md:
empty, multilingual UTF-8, binary-mixed, and context-spanning inputs across the
TinyLlama Q8_0/Q4_0, Qwen2.5 Q8_0, and SmolLM2 Q8_0 GGUFs.
USAGE
}

bin="target/release/detllm"
out_dir="/tmp/detllm-roundtrip-matrix"
threads="8"
n_ctx="8"
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
    --threads)
      threads="${2:?missing value for --threads}"
      shift 2
      ;;
    --n-ctx)
      n_ctx="${2:?missing value for --n-ctx}"
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

rm -rf "$out_dir"
mkdir -p "$out_dir"

: > "$out_dir/empty.bin"
printf '%b' 'detllm multilingual validation\nHello, \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf, \xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7, \xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82, \xce\x9a\xce\xb1\xce\xbb\xce\xb7\xce\xbc\xce\xad\xcf\x81\xce\xb1, \xf0\x9f\x98\x80\n' > "$out_dir/multilingual.txt"
printf '%b' 'detllm\0binary\xff\xc0\x04\nvalidation\n' > "$out_dir/binary-mixed.bin"
printf 'one two three four five six seven eight nine ten eleven twelve.\n' > "$out_dir/context-spanning.txt"

echo "== inputs =="
sha256sum "$out_dir/empty.bin" "$out_dir/multilingual.txt" "$out_dir/binary-mixed.bin" "$out_dir/context-spanning.txt"
wc -c "$out_dir/empty.bin" "$out_dir/multilingual.txt" "$out_dir/binary-mixed.bin" "$out_dir/context-spanning.txt"

models=(
  "tinyllama-q8:$tinyllama_q8"
  "tinyllama-q4:$tinyllama_q4"
  "qwen25-q8:$qwen25_q8"
  "smollm2-q8:$smollm2_q8"
)
inputs=(
  "empty:$out_dir/empty.bin"
  "multilingual:$out_dir/multilingual.txt"
  "binary-mixed:$out_dir/binary-mixed.bin"
  "context-spanning:$out_dir/context-spanning.txt"
)

for model_entry in "${models[@]}"; do
  model_name="${model_entry%%:*}"
  model_path="${model_entry#*:}"
  for input_entry in "${inputs[@]}"; do
    input_name="${input_entry%%:*}"
    input_path="${input_entry#*:}"
    out_base="$out_dir/${model_name}-${input_name}"
    echo "== $model_name $input_name =="
    "$bin" compress -m "$model_path" -i "$input_path" -o "$out_base.dtlz" --n-ctx "$n_ctx" --threads "$threads"
    "$bin" decompress -m "$model_path" -i "$out_base.dtlz" -o "$out_base.restored" --threads "$threads"
    cmp "$input_path" "$out_base.restored"
    sha256sum "$out_base.dtlz" "$out_base.restored"
    wc -c "$out_base.dtlz" "$out_base.restored"
  done
done
