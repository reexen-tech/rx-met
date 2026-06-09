#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_CPP_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${MODEL:-$LLAMA_CPP_ROOT/models/shared_models/Qwen/Qwen3-30B-A3B-Instruct-2507/Qwen3-30B-A3B-Instruct-2507-Q4_K.gguf}"
BUILD_DIR="${BUILD_DIR:-$LLAMA_CPP_ROOT/build}"
EVAL_BIN="${EVAL_BIN:-$BUILD_DIR/bin/eval_single_token}"
TARGET_LAYER="${TARGET_LAYER:-0}"
TOKENS="${TOKENS:-2 4 8 16 32}"
N_THREADS="${N_THREADS:-4}"
DUMP_BASE="${DUMP_BASE:-$SCRIPT_DIR/dump_reex_cuda_trace_sweep}"

if [[ ! -x "$EVAL_BIN" ]]; then
    ALT_BIN="$BUILD_DIR/tests/eval_single_token"
    if [[ -x "$ALT_BIN" ]]; then
        EVAL_BIN="$ALT_BIN"
    else
        echo "ERROR: eval_single_token not found: $EVAL_BIN"
        exit 1
    fi
fi

if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: model not found: $MODEL"
    exit 1
fi

mkdir -p "$DUMP_BASE"

make_prompt() {
    python3 - "$1" <<'PY'
import sys
n = int(sys.argv[1])
print(" ".join(["a"] * n))
PY
}

parse_log_field() {
    local file="$1"
    local pattern="$2"
    python3 - "$file" "$pattern" <<'PY'
import re
import sys

text = open(sys.argv[1], "r", encoding="utf-8", errors="ignore").read()
match = re.search(sys.argv[2], text)
print(match.group(1) if match else "")
PY
}

echo "=================================================================="
echo "  REEX CUDA trace token sweep"
echo "=================================================================="
echo "  MODEL=$MODEL"
echo "  EVAL_BIN=$EVAL_BIN"
echo "  TARGET_LAYER=$TARGET_LAYER"
echo "  TOKENS=$TOKENS"
echo "  DUMP_BASE=$DUMP_BASE"
echo ""

printf "| requested_tokens | actual_tokens | q8_mul_mat_hits | q8_mul_mat_id_hits | q16_mul_mat_hits | q16_batch_fallbacks | q16_mul_mat_id_hits | q16_mul_mat_id_unsupported |\n"
printf "|-----------------:|-------------:|----------------:|-------------------:|------------------:|--------------------:|--------------------:|---------------------------:|\n"

for requested in $TOKENS; do
    prompt="$(make_prompt "$requested")"
    dump_dir="$DUMP_BASE/tok_${requested}"
    log_file="$dump_dir/run.log"
    trace_file="$dump_dir/reex_cuda_trace.txt"

    rm -rf "$dump_dir"
    mkdir -p "$dump_dir"

    env REEX_DUMP_DIR="$dump_dir" \
        REEX_DUMP_LAYER="$TARGET_LAYER" \
        REEX_TARGET_LAYER="$TARGET_LAYER" \
        REEX_PRINT_CUDA_TRACE=1 \
        "$EVAL_BIN" -m "$MODEL" -t "$N_THREADS" -p "$prompt" \
        > /dev/null 2> "$log_file"

    actual_tokens="$(parse_log_field "$log_file" 'Prompt: ".*" \(([0-9]+) tokens\)')"
    q8_mul_mat_hits="$(parse_log_field "$trace_file" 'q8_mul_mat_hits=([0-9]+)')"
    q8_mul_mat_id_hits="$(parse_log_field "$trace_file" 'q8_mul_mat_id_hits=([0-9]+)')"
    q16_mul_mat_hits="$(parse_log_field "$trace_file" 'q16_mul_mat_hits=([0-9]+)')"
    q16_batch_fallbacks="$(parse_log_field "$trace_file" 'q16_batch_fallbacks=([0-9]+)')"
    q16_mul_mat_id_hits="$(parse_log_field "$trace_file" 'q16_mul_mat_id_hits=([0-9]+)')"
    q16_mul_mat_id_unsupported="$(parse_log_field "$trace_file" 'q16_mul_mat_id_unsupported=([0-9]+)')"

    printf "| %16s | %12s | %15s | %18s | %17s | %19s | %19s | %26s |\n" \
        "$requested" \
        "${actual_tokens:-N/A}" \
        "${q8_mul_mat_hits:-N/A}" \
        "${q8_mul_mat_id_hits:-N/A}" \
        "${q16_mul_mat_hits:-N/A}" \
        "${q16_batch_fallbacks:-N/A}" \
        "${q16_mul_mat_id_hits:-N/A}" \
        "${q16_mul_mat_id_unsupported:-N/A}"
done
