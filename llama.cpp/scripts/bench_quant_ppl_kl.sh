#!/usr/bin/env bash
#
# Compare one or more quantized models against an F16/BF16 baseline using
# llama-perplexity PPL and KL divergence.
#
# Example (GPTQ + RTN):
#   scripts/bench_quant_ppl_kl.sh \
#     --base /models/model-f16.gguf \
#     --model rtn=/models/model-rtn.gguf \
#     --model gptq=/models/model-gptq.gguf \
#     --dataset /data/wiki.test.raw \
#     --output results/quant-benchmark
#
# Modes:
#   all  - baseline PPL/logits, candidate PPL, candidate KL (default)
#   base - baseline PPL and logits only
#   ppl  - baseline and candidate PPL only
#   kl   - candidate KL only; reuses OUTPUT/f16-logits.bin
#
# Environment:
#   CUDA_DEVICE=0
#   BUILD_DIR=build_cuda_q64_noreex
#   CTX=512
#   CHUNKS=8
#   NGL=99

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bench_quant_ppl_kl.sh \
    --base BASE_GGUF \
    --model TAG=MODEL_GGUF [--model TAG=MODEL_GGUF ...] \
    --dataset DATASET \
    --output OUTPUT_DIR \
    [--mode all|base|ppl|kl]
EOF
}

BASE_MODEL=
DATASET=
OUTPUT_DIR=
MODE=all
MODEL_TAGS=()
MODEL_PATHS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base)
            BASE_MODEL=${2:?missing value for --base}
            shift 2
            ;;
        --model)
            spec=${2:?missing value for --model}
            if [[ "$spec" != *=* ]]; then
                echo "[ERROR] --model must use TAG=PATH: $spec" >&2
                exit 2
            fi
            tag=${spec%%=*}
            path=${spec#*=}
            if [[ -z "$tag" || -z "$path" || ! "$tag" =~ ^[A-Za-z0-9_.-]+$ ]]; then
                echo "[ERROR] invalid --model TAG=PATH: $spec" >&2
                exit 2
            fi
            MODEL_TAGS+=("$tag")
            MODEL_PATHS+=("$path")
            shift 2
            ;;
        --dataset)
            DATASET=${2:?missing value for --dataset}
            shift 2
            ;;
        --output)
            OUTPUT_DIR=${2:?missing value for --output}
            shift 2
            ;;
        --mode)
            MODE=${2:?missing value for --mode}
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$BASE_MODEL" || -z "$DATASET" || -z "$OUTPUT_DIR" ]]; then
    usage >&2
    exit 2
fi
if [[ ${#MODEL_TAGS[@]} -eq 0 && "$MODE" != "base" ]]; then
    echo "[ERROR] at least one --model TAG=PATH is required" >&2
    exit 2
fi
if [[ "$MODE" != "all" && "$MODE" != "base" && "$MODE" != "ppl" && "$MODE" != "kl" ]]; then
    echo "[ERROR] unsupported mode: $MODE" >&2
    exit 2
fi

CUDA_DEVICE=${CUDA_DEVICE:-0}
BUILD_DIR=${BUILD_DIR:-build_cuda_q64_noreex}
CTX=${CTX:-512}
CHUNKS=${CHUNKS:-8}
NGL=${NGL:-99}

PPL_BIN="${BUILD_DIR}/bin/llama-perplexity"
BASE_LOGITS="${OUTPUT_DIR}/f16-logits.bin"

for path in "$BASE_MODEL" "$DATASET" "$PPL_BIN" "${MODEL_PATHS[@]}"; do
    [[ -e "$path" ]] || {
        echo "[ERROR] missing: $path" >&2
        exit 1
    }
done

mkdir -p "$OUTPUT_DIR"

run_ppl() {
    local tag=$1
    local model=$2
    shift 2
    echo "[$tag] PPL (ctx=$CTX chunks=$CHUNKS gpu=$CUDA_DEVICE)"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PPL_BIN" \
        -m "$model" -f "$DATASET" \
        -ngl "$NGL" -c "$CTX" --chunks "$CHUNKS" \
        "$@" 2>&1 | tee "${OUTPUT_DIR}/ppl-${tag}.log"
}

run_kl() {
    local tag=$1
    local model=$2
    [[ -e "$BASE_LOGITS" ]] || {
        echo "[ERROR] missing baseline logits: $BASE_LOGITS" >&2
        echo "Run with --mode base or --mode all first." >&2
        exit 1
    }
    echo "[$tag] KL vs baseline (ctx=$CTX chunks=$CHUNKS gpu=$CUDA_DEVICE)"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PPL_BIN" \
        -m "$model" -f "$DATASET" \
        -ngl "$NGL" -c "$CTX" --chunks "$CHUNKS" \
        --kl-divergence --kl-divergence-base "$BASE_LOGITS" \
        2>&1 | tee "${OUTPUT_DIR}/kl-${tag}.log"
}

if [[ "$MODE" == "all" || "$MODE" == "base" ]]; then
    run_ppl baseline "$BASE_MODEL" --save-all-logits "$BASE_LOGITS"
elif [[ "$MODE" == "ppl" ]]; then
    run_ppl baseline "$BASE_MODEL"
fi

if [[ "$MODE" == "all" || "$MODE" == "ppl" ]]; then
    for index in "${!MODEL_TAGS[@]}"; do
        run_ppl "${MODEL_TAGS[$index]}" "${MODEL_PATHS[$index]}"
    done
fi

if [[ "$MODE" == "all" || "$MODE" == "kl" ]]; then
    for index in "${!MODEL_TAGS[@]}"; do
        run_kl "${MODEL_TAGS[$index]}" "${MODEL_PATHS[$index]}"
    done
fi

echo
echo "========== SUMMARY =========="
for log in "$OUTPUT_DIR"/ppl-*.log; do
    [[ -e "$log" ]] || continue
    tag=${log##*/ppl-}
    tag=${tag%.log}
    value=$(grep -oE "Final estimate: PPL = [0-9.]+( \\+/- [0-9.]+)?" "$log" | tail -n 1 || true)
    printf "%-16s %s\n" "ppl/$tag" "${value:-N/A}"
done
for log in "$OUTPUT_DIR"/kl-*.log; do
    [[ -e "$log" ]] || continue
    tag=${log##*/kl-}
    tag=${tag%.log}
    value=$(grep -oE "Mean +KLD: +[0-9.]+" "$log" | tail -n 1 || true)
    printf "%-16s %s\n" "kl/$tag" "${value:-N/A}"
done
