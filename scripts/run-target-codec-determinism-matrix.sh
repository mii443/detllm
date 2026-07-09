#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: scripts/run-target-codec-determinism-matrix.sh --tinyllama-q8 PATH --tinyllama-q4 PATH --qwen25-q8 PATH --smollm2-q8 PATH [--bin target/release/detllm] [--extra-bin LABEL=PATH ...] [--out DIR] [--n-ctx N]

Checks target-model DTLZ payload determinism across thread-count settings. The
matrix compresses byte-escape and context-rollover inputs with
threads={1,2,7,16}, requires the DTLZ SHA-256 to match bit-for-bit for each
model/input pair, and decompresses every output back to the original bytes.
Additional labeled binaries can be supplied with --extra-bin to compare
scalar/simd/parallel builds in the same matrix.
USAGE
}

bin="target/release/detllm"
extra_bin_labels=()
extra_bin_paths=()
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
    --extra-bin)
      spec="${2:?missing value for --extra-bin}"
      if [[ "$spec" != *=* ]]; then
        echo "--extra-bin must be LABEL=PATH" >&2
        exit 2
      fi
      label="${spec%%=*}"
      path="${spec#*=}"
      if [[ -z "$label" || -z "$path" ]]; then
        echo "--extra-bin must be LABEL=PATH with non-empty LABEL and PATH" >&2
        exit 2
      fi
      extra_bin_labels+=("$label")
      extra_bin_paths+=("$path")
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

for path in "$bin" "${extra_bin_paths[@]}" "$tinyllama_q8" "$tinyllama_q4" "$qwen25_q8" "$smollm2_q8"; do
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
bin_labels=("primary" "${extra_bin_labels[@]}")
bin_paths=("$bin" "${extra_bin_paths[@]}")
multi_bin="false"
if [[ "${#bin_paths[@]}" -gt 1 ]]; then
  multi_bin="true"
fi

for model_entry in "${models[@]}"; do
  model_name="${model_entry%%:*}"
  model_path="${model_entry#*:}"
  for input_entry in "${inputs[@]}"; do
    input_name="${input_entry%%:*}"
    input_path="${input_entry#*:}"
    baseline_hash=""
    baseline_size=""
    echo "== $model_name $input_name =="
    for bin_index in "${!bin_paths[@]}"; do
      bin_label="${bin_labels[$bin_index]}"
      bin_path="${bin_paths[$bin_index]}"
      for threads in 1 2 7 16; do
        out_base="$out_dir/${model_name}-${input_name}-${bin_label}-threads${threads}"
        "$bin_path" compress -m "$model_path" -i "$input_path" -o "$out_base.dtlz" --n-ctx "$n_ctx" --threads "$threads"
        "$bin_path" decompress -m "$model_path" -i "$out_base.dtlz" -o "$out_base.restored" --threads "$threads"
        cmp "$input_path" "$out_base.restored"

        dtlz_hash="$(sha256sum "$out_base.dtlz" | awk '{print $1}')"
        restored_hash="$(sha256sum "$out_base.restored" | awk '{print $1}')"
        dtlz_size="$(wc -c < "$out_base.dtlz")"
        restored_size="$(wc -c < "$out_base.restored")"

        if [[ -z "$baseline_hash" ]]; then
          baseline_hash="$dtlz_hash"
          baseline_size="$dtlz_size"
        elif [[ "$dtlz_hash" != "$baseline_hash" ]]; then
          echo "DTLZ hash mismatch for $model_name $input_name bin=$bin_label threads=$threads: got $dtlz_hash expected $baseline_hash" >&2
          exit 1
        elif [[ "$dtlz_size" != "$baseline_size" ]]; then
          echo "DTLZ size mismatch for $model_name $input_name bin=$bin_label threads=$threads: got $dtlz_size expected $baseline_size" >&2
          exit 1
        fi

        if [[ "$multi_bin" == "true" ]]; then
          echo "codec model=$model_name input=$input_name bin=$bin_label threads=$threads dtlz_bytes=$dtlz_size dtlz_sha256=$dtlz_hash restored_bytes=$restored_size restored_sha256=$restored_hash"
        else
          echo "codec model=$model_name input=$input_name threads=$threads dtlz_bytes=$dtlz_size dtlz_sha256=$dtlz_hash restored_bytes=$restored_size restored_sha256=$restored_hash"
        fi
      done
    done
    echo "ok model=$model_name input=$input_name dtlz_bytes=$baseline_size dtlz_sha256=$baseline_hash"
  done
done
