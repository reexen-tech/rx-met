#!/usr/bin/env bash
# Block-64 PPL sweep：wikitext-103 validation，与历史 baseline 同口径（ctx=512, chunks=-1）。
#
#   REEX_Q64_PSUM_BITS=8 ./run_qwen35_block64_ppl_sweep.sh all
#   ./run_qwen35_block64_ppl_sweep.sh status
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="$SCRIPT_DIR/run_qwen35_block64_gpu_docker.sh"
CONTAINER="${BLOCK64_CONTAINER:-quant-gru-cuda128}"
WORKDIR="${BLOCK64_WORKDIR:-/mnt/data8t/zcx/aimet_rx/llama.cpp}"
OUT="${BLOCK64_OUT:-/mnt/data8t/zcx/aimet_rx/llama.cpp/tests/test-reex/results/qwen35b_a3b_block64_tensor/ppl}"
LOG_DIR="${PPL_LOG_DIR:-/tmp/qwen35_block64_ppl_logs}"
# B=8 七组 + Q4_0_64 B=off
PPL_TYPES="${PPL_TYPES:-Q4_0_64 Q8_0_64 Q2_K_64 Q3_K_64 Q4_K_64 Q5_K_64 Q6_K_64}"
PSUM_B="${REEX_Q64_PSUM_BITS:-8}"

ngl_for_type() {
    case "$1" in
        Q8_0_64) echo "${SWEEP_NGL_TEST_Q8:-8}" ;;
        Q6_K_64) echo "${SWEEP_NGL_TEST_Q6:-48}" ;;
        *) echo "${SWEEP_NGL_TEST:-99}" ;;
    esac
}

log() { echo "[$(date '+%F %T')] $*"; }

ppl_json() {
    local t="$1" b="$2"
    echo "$OUT/perplexity_Qwen3.5-35B-A3B-${t}_B${b}_validation.json"
}

ppl_log() {
    local t="$1" b="$2"
    echo "$LOG_DIR/ppl_${t}_B${b}.log"
}

run_one() {
    local t="$1" b="$2"
    local jf lf
    jf="$(ppl_json "$t" "$b")"
    lf="$(ppl_log "$t" "$b")"
    if [[ -s "$jf" ]] && grep -q '"ppl"' "$jf" 2>/dev/null; then
        log "SKIP $t B=$b (json exists)"
        return 0
    fi
    mkdir -p "$OUT" "$LOG_DIR"
    local ngl
    ngl="$(ngl_for_type "$t")"
    log "START PPL $t B=$b ngl=$ngl"
    BLOCK64_TYPE="$t" REEX_Q64_PSUM_BITS="$b" BLOCK64_NGL_TEST="$ngl" \
        "$RUN" ppl >"$lf" 2>&1
    docker exec "$CONTAINER" python3 \
        "$WORKDIR/tests/test-reex/parse_ppl_output.py" \
        "$lf" "Qwen3.5-35B-A3B-${t}.gguf" "$jf" 2>/dev/null || \
    python3 "$SCRIPT_DIR/parse_ppl_output.py" "$lf" \
        "Qwen3.5-35B-A3B-${t}.gguf" "$jf"
    log "DONE $t B=$b"
}

run_all() {
    for t in $PPL_TYPES; do
        run_one "$t" "$PSUM_B"
    done
    if [[ "$PSUM_B" != "off" ]]; then
        run_one "Q4_0_64" "off"
    fi
}

show_status() {
    echo "=== PPL status (OUT=$OUT, B default=$PSUM_B) ==="
    printf "  %-10s %-6s %-8s %s\n" "TYPE" "B" "JSON" "ppl"
    for t in $PPL_TYPES; do
        for b in "$PSUM_B" off; do
            [[ "$b" == off && "$t" != Q4_0_64 ]] && continue
            jf="$(ppl_json "$t" "$b")"
            if [[ -s "$jf" ]]; then
                ppl=$(python3 -c "import json; print(json.load(open('$jf'))['result'].get('ppl','?'))" 2>/dev/null || echo ERR)
                printf "  %-10s %-6s %-8s %s\n" "$t" "$b" OK "$ppl"
            else
                printf "  %-10s %-6s %-8s %s\n" "$t" "$b" MISSING —
            fi
        done
    done
}

case "${1:-}" in
    all) run_all ;;
    status) show_status ;;
    *)
        echo "Usage: $0 <all|status>"
        exit 2
        ;;
esac
