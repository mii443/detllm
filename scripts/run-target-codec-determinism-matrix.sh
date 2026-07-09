#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: scripts/run-target-codec-determinism-matrix.sh --tinyllama-q8 PATH --tinyllama-q4 PATH --qwen25-q8 PATH --smollm2-q8 PATH [--bin target/release/detllm] [--out DIR] [--n-ctx N]

Checks target-model DTLZ payload determinism across thread-count settings. The
matrix compresses byte-escape and context-rollover inputs with
threads={1,2,7,16}, requires the DTLZ SHA-256 to match bit-for-bit for each
model/input pair, and decompresses every output back to the original bytes.
USAGE
}

bin="target/release/detllm"
out_dir="/tmp/detllm-codec-determinism-matrix"
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

printf '%b' 'detllm\0binary\xff\xc0\x04\nvalidation\n' > "$out_dir/binary-mixed.bin"
printf 'one two three four five six seven eight nine ten eleven twelve.\n' > "$out_dir/context-spanning.txt"

echo "== inputs =="
sha256sum "$out_dir/binary-mixed.bin" "$out_dir/context-spanning.txt"
wc -c "$out_dir/binary-mixed.bin" "$out_dir/context-spanning.txt"

models=(
  "tinyllama-q8:$tinyllama_q8"
  "tinyllama-q4:$tinyllama_q4"
  "qwen25-q8:$qwen25_q8"
  "smollm2-q8:$smollm2_q8"
)
inputs=(
  "binary-mixed:$out_dir/binary-mixed.bin"
  "context-spanning:$out_dir/context-spanning.txt"
)

for model_entry in "${models[@]}"; do
  model_name="${model_entry%%:*}"
  model_path="${model_entry#*:}"
  for input_entry in "${inputs[@]}"; do
    input_name="${input_entry%%:*}"
    input_path="${input_entry#*:}"
    baseline_hash=""
    baseline_size=""
    echo "== $model_name $input_name =="
    for threads in 1 2 7 16; do
      out_base="$out_dir/${model_name}-${input_name}-threads${threads}"
      "$bin" compress -m "$model_path" -i "$input_path" -o "$out_base.dtlz" --n-ctx "$n_ctx" --threads "$threads"
      "$bin" decompress -m "$model_path" -i "$out_base.dtlz" -o "$out_base.restored" --threads "$threads"
      cmp "$input_path" "$out_base.restored"

      dtlz_hash="$(sha256sum "$out_base.dtlz" | awk '{print $1}')"
      restored_hash="$(sha256sum "$out_base.restored" | awk '{print $1}')"
      dtlz_size="$(wc -c < "$out_base.dtlz")"
      restored_size="$(wc -c < "$out_base.restored")"

      if [[ -z "$baseline_hash" ]]; then
        baseline_hash="$dtlz_hash"
        baseline_size="$dtlz_size"
      elif [[ "$dtlz_hash" != "$baseline_hash" ]]; then
        echo "DTLZ hash mismatch for $model_name $input_name threads=$threads: got $dtlz_hash expected $baseline_hash" >&2
        exit 1
      elif [[ "$dtlz_size" != "$baseline_size" ]]; then
        echo "DTLZ size mismatch for $model_name $input_name threads=$threads: got $dtlz_size expected $baseline_size" >&2
        exit 1
      fi

      echo "codec model=$model_name input=$input_name threads=$threads dtlz_bytes=$dtlz_size dtlz_sha256=$dtlz_hash restored_bytes=$restored_size restored_sha256=$restored_hash"
    done
    echo "ok model=$model_name input=$input_name dtlz_bytes=$baseline_size dtlz_sha256=$baseline_hash"
  done
done
