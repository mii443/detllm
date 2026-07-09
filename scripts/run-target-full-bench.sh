#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: scripts/run-target-full-bench.sh --model PATH --input /tmp/enwik8 [--out DIR] [--name NAME] [--limit-bytes N] [--limit-tokens N] [--n-ctx N] [--threads N] [--progress-every N] [--verify-dtlz PATH] [--encode-only] [--estimate-full-run] [--warmup]

Runs a single target-model bench-file measurement with durable output files.
By default this is the final enwik8 first-1MB acceptance shape: no token limit,
round-trip verification, no warmup, phase output, progress output, and a
bench-file summary plus DTLZ output written through xtask's atomic output
paths. A progress summary file is atomically updated while the benchmark is
running.
USAGE
}

model=""
input=""
out_dir="/tmp/detllm-target-bench"
name=""
limit_bytes="1048576"
limit_tokens=""
n_ctx="2048"
threads="8"
progress_every="1000"
verify_dtlz=""
encode_only="false"
estimate_full_run="false"
warmup="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      model="${2:?missing value for --model}"
      shift 2
      ;;
    --input)
      input="${2:?missing value for --input}"
      shift 2
      ;;
    --out)
      out_dir="${2:?missing value for --out}"
      shift 2
      ;;
    --name)
      name="${2:?missing value for --name}"
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
    --verify-dtlz)
      verify_dtlz="${2:?missing value for --verify-dtlz}"
      shift 2
      ;;
    --encode-only)
      encode_only="true"
      shift
      ;;
    --estimate-full-run)
      estimate_full_run="true"
      shift
      ;;
    --warmup)
      warmup="true"
      shift
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

if [[ -z "$model" || -z "$input" ]]; then
  usage >&2
  exit 2
fi

for path in "$model" "$input"; do
  if [[ ! -f "$path" ]]; then
    echo "missing file: $path" >&2
    exit 1
  fi
done

if [[ "$estimate_full_run" == "true" && -z "$limit_tokens" ]]; then
  echo "--estimate-full-run requires --limit-tokens" >&2
  exit 2
fi

if [[ -n "$verify_dtlz" && ! -f "$verify_dtlz" ]]; then
  echo "missing file: $verify_dtlz" >&2
  exit 1
fi

if [[ -n "$verify_dtlz" && "$encode_only" == "true" ]]; then
  echo "--verify-dtlz cannot be combined with --encode-only" >&2
  exit 2
fi

if [[ -n "$verify_dtlz" && "$estimate_full_run" == "true" ]]; then
  echo "--verify-dtlz cannot be combined with --estimate-full-run" >&2
  exit 2
fi

if [[ -z "$name" ]]; then
  name="$(basename "$model")"
  name="${name%.gguf}"
  name="${name//[^A-Za-z0-9._-]/_}"
fi

mkdir -p "$out_dir"
summary_path="$out_dir/$name.summary"
progress_summary_path="$out_dir/$name.progress"
dtlz_path="$out_dir/$name.dtlz"
log_path="$out_dir/$name.log"

cmd=(
  cargo run --release -p xtask --features parallel,simd -- bench-file
  --model "$model"
  --input "$input"
  --limit-bytes "$limit_bytes"
  --n-ctx "$n_ctx"
  --threads "$threads"
  --iters 1
  --show-phases
  --summary "$summary_path"
  --progress-every "$progress_every"
  --progress-summary "$progress_summary_path"
)

if [[ -n "$verify_dtlz" ]]; then
  cmd+=(--verify-dtlz "$verify_dtlz")
else
  cmd+=(--output-dtlz "$dtlz_path")
fi

if [[ -n "$limit_tokens" ]]; then
  cmd+=(--limit-tokens "$limit_tokens")
fi

if [[ "$warmup" != "true" ]]; then
  cmd+=(--no-warmup)
fi

if [[ "$encode_only" == "true" ]]; then
  cmd+=(--encode-only)
fi

if [[ "$estimate_full_run" == "true" ]]; then
  cmd+=(--estimate-full-run)
fi

echo "summary: $summary_path"
echo "progress-summary: $progress_summary_path"
if [[ -n "$verify_dtlz" ]]; then
  echo "verify-dtlz: $verify_dtlz"
else
  echo "dtlz: $dtlz_path"
fi
echo "log: $log_path"
printf 'command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}" 2>&1 | tee "$log_path"
