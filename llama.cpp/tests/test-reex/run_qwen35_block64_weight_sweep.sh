#!/usr/bin/env bash
# 并行推进 §4.2.2 剩余 6 组 weight sweep（Q4_0_64 已完成）。
#
# 策略（单 GPU 容器）：
#   Phase A — llama-quantize：CPU 并行（默认 3 路），生成 6 个 GGUF
#   Phase B — dump-test-only：GPU 串行（一把锁），复用已有 F16 Gold
#   Phase C — compare：CPU，dump 完一组立即 compare（可与下一组 dump 重叠）
#
# 用法：
#   ./run_qwen35_block64_weight_sweep.sh quantize-all          # 仅量化 6 个 GGUF
#   ./run_qwen35_block64_weight_sweep.sh dump-remaining        # 已有 GGUF 的 type 串行 dump+compare
#   ./run_qwen35_block64_weight_sweep.sh all                   # A 完成后自动 B+C
#   ./run_qwen35_block64_weight_sweep.sh status                # 查看进度
#
# 环境变量：
#   SWEEP_TYPES='Q8_0_64 Q2_K_64 ...'   默认 6 种待做 type
#   SWEEP_QUANT_JOBS=3                  并行量化路数
#   REEX_Q64_PSUM_BITS=8                Test dump 用 B=8（与 Q4_0_64 定档一致）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="$SCRIPT_DIR/run_qwen35_block64_gpu_docker.sh"
OUT="${BLOCK64_OUT:-/mnt/data8t/zcx/aimet_rx/llama.cpp/tests/test-reex/results/qwen35b_a3b_block64_tensor}"
MODEL_DIR="${SWEEP_MODEL_DIR:-/mnt/data8t/share/models/Qwen/Qwen3.5-35B-A3B}"
MODEL_F16="${MODEL_F16:-$MODEL_DIR/Qwen3.5-35B-A3B-f16.gguf}"
SWEEP_TYPES="${SWEEP_TYPES:-Q8_0_64 Q2_K_64 Q3_K_64 Q4_K_64 Q5_K_64 Q6_K_64}"
SWEEP_QUANT_JOBS="${SWEEP_QUANT_JOBS:-3}"
PSUM_B="${REEX_Q64_PSUM_BITS:-8}"
LOG_DIR="${SWEEP_LOG_DIR:-/tmp/qwen35_block64_sweep_logs}"

model_path() {
    local t="$1"
    echo "$MODEL_DIR/Qwen3.5-35B-A3B-${t}.gguf"
}

log() { echo "[$(date '+%F %T')] $*"; }

ensure_dirs() {
    mkdir -p "$LOG_DIR"
}

quantize_one() {
    local t="$1"
    local out
    out="$(model_path "$t")"
    local logf="$LOG_DIR/quantize_${t}.log"
    if [[ -s "$out" ]]; then
        log "SKIP quantize $t (exists $(du -h "$out" | awk '{print $1}'))"
        return 0
    fi
    log "START quantize $t -> $out"
    BLOCK64_TYPE="$t" MODEL_Q64="$out" REEX_Q64_PSUM_BITS="$PSUM_B" \
        "$RUN" quantize >"$logf" 2>&1
    log "DONE quantize $t"
}

quantize_all_parallel() {
    ensure_dirs
    log "Phase A: parallel quantize (jobs=$SWEEP_QUANT_JOBS) types=$SWEEP_TYPES"
    local t running=0 ec=0
    for t in $SWEEP_TYPES; do
        while (( running >= SWEEP_QUANT_JOBS )); do
            if ! wait -n; then ec=1; fi
            running=$((running - 1))
        done
        quantize_one "$t" &
        running=$((running + 1))
        log "queued quantize $t (running=$running)"
    done
    while (( running > 0 )); do
        if ! wait -n; then ec=1; fi
        running=$((running - 1))
    done
    return "$ec"
}

dump_compare_one() {
    local t="$1"
    local mp logf cmp dump_dir nbins=0
    mp="$(model_path "$t")"
    logf="$LOG_DIR/dump_${t}.log"
    cmp="$OUT/prefill8192_${t}_B${PSUM_B}_compare.log"
    dump_dir="$OUT/dump_${t}_B${PSUM_B}_prefill8192"
    if [[ -s "$cmp" ]]; then
        log "SKIP $t (compare done)"
        return 0
    fi
    if [[ ! -s "$mp" ]]; then
        log "SKIP dump $t (missing GGUF $mp)"
        return 1
    fi
    if [[ -d "$dump_dir" ]]; then
        nbins=$(find "$dump_dir" -maxdepth 1 -name '*.bin' 2>/dev/null | wc -l)
    fi
    if (( nbins >= 1000 )); then
        log "SKIP dump $t (reuse $nbins bins)"
    else
        log "START dump-test $t (B=$PSUM_B)"
        BLOCK64_TYPE="$t" MODEL_Q64="$mp" REEX_Q64_PSUM_BITS="$PSUM_B" \
            "$RUN" dump-test-only >"$logf" 2>&1
    fi
    log "START compare $t"
    BLOCK64_TYPE="$t" REEX_Q64_PSUM_BITS="$PSUM_B" \
        "$RUN" compare >"$LOG_DIR/compare_${t}.log" 2>&1
    cp -f "$LOG_DIR/compare_${t}.log" "$cmp" 2>/dev/null || true
    log "DONE $t"
}

dump_remaining_serial() {
    ensure_dirs
    log "Phase B+C: serial GPU dump + compare (single GPU)"
    local ec=0
    for t in $SWEEP_TYPES; do
        dump_compare_one "$t" || ec=1
    done
    return "$ec"
}

show_status() {
    echo "=== sweep status (OUT=$OUT) ==="
    for t in Q4_0_64 $SWEEP_TYPES; do
        mp="$(model_path "$t")"
        gguf="MISSING"
        [[ -s "$mp" ]] && gguf="OK $(du -h "$mp" | awk '{print $1}')"
        dump="MISSING"
        [[ -d "$OUT/dump_${t}_B${PSUM_B}_prefill8192" ]] && dump="OK"
        cmp="MISSING"
        [[ -s "$OUT/prefill8192_${t}_B${PSUM_B}_compare.log" ]] && cmp="OK"
        printf "  %-10s GGUF=%-12s dump=%-6s compare=%s\n" "$t" "$gguf" "$dump" "$cmp"
    done
    echo "logs: $LOG_DIR"
}

usage() {
    cat <<EOF
Usage: $0 <quantize-all|dump-remaining|all|status>

  quantize-all     并行量化 6 个 GGUF（默认 3 路）
  dump-remaining   串行 GPU dump-test + compare（复用 F16 Gold）
  all              quantize-all 成功后 dump-remaining
  status           查看各 type GGUF/dump/compare 状态
EOF
}

cmd="${1:-}"
case "$cmd" in
    quantize-all) quantize_all_parallel ;;
    dump-remaining) dump_remaining_serial ;;
    all)
        quantize_all_parallel
        dump_remaining_serial
        ;;
    status) show_status ;;
    *) usage; exit 2 ;;
esac
