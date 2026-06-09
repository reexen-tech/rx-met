#!/usr/bin/env bash
# 自研 REEX 验证总入口：先「官网基线」再「自研多变体矩阵」。
# 日常更推荐从 ./scripts/reex_validate.sh 进入。
# 完整说明见:
#   - tests/test-reex/REEX_FULL_VALIDATION.md
#   - scripts/REEX_VALIDATION_GUIDE.md
#
# 用法（在 llama.cpp 根目录）:
#   ./scripts/run_reex_full_validation.sh
#
# 环境变量:
#   VERIFY_ROOT            仓库根目录（默认本脚本/../）
#   VERIFY_JOBS            并行编译线程（默认 nproc）
#   SKIP_CUDA=1            跳过 Tier0/Tier1 中所有 CUDA
#   SKIP_TIER1=1           只跑 Tier0（上游基线）
#   SKIP_TIER0=1           只跑 Tier1（verify_build_variants）
#   VERIFY_QUICK=1         传给 verify_build_variants：缩小 REEX 矩阵
#   VERIFY_GEMM_Q8=1       传给 verify_build_variants：增加 Q8 变体
#   VERIFY_SKIP_RUNTIME=1  传给 verify_build_variants：编译后不跑二进制冒烟
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERIFY_ROOT="${VERIFY_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VERIFY_JOBS="${VERIFY_JOBS:-${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc 2>/dev/null || echo 4)}}"
VERIFY_CMAKE_GENERATOR="${VERIFY_CMAKE_GENERATOR:-}"
REEX_GATE_PROFILE="${REEX_GATE_PROFILE:-standalone}"
REEX_RESUME="${REEX_RESUME:-1}"
VERIFY_CLEAN_BUILDS="${VERIFY_CLEAN_BUILDS:-0}"
VALIDATION_STATE_DIR="${REEX_FULL_VALIDATION_STATE_DIR:-$VERIFY_ROOT/.reex-validation/full-validation-${REEX_GATE_PROFILE}}"
if [[ -z "${VERIFY_TIER0_CUDA_OPS:-}" && "$REEX_GATE_PROFILE" == "full" ]]; then
  # Focus Tier0b on representative upstream CUDA ops that cover the modified REEX-sensitive paths.
  VERIFY_TIER0_CUDA_OPS="MUL_MAT,MUL_MAT_ID,MUL_MAT_ID_FUSION,MUL_MAT_VEC_FUSION,SET_ROWS,ROPE_SET_ROWS,RMS_NORM_MUL_ROPE,SOFT_MAX,FLASH_ATTN_EXT,TOPK_MOE,SSM_SCAN,GELU_QUICK"
fi
cd "$VERIFY_ROOT"
mkdir -p "$VALIDATION_STATE_DIR"

# 宿主机若未把 /usr/local/cuda/bin 加入 PATH，在此补全（容器内通常已配置）
# shellcheck source=/dev/null
source "$SCRIPT_DIR/reex_cuda_env.sh"
reex_prepend_cuda_path || true

OPS_BIN=""
format_duration() {
  local total_sec="${1:-0}"
  local h=$(( total_sec / 3600 ))
  local m=$(( (total_sec % 3600) / 60 ))
  local s=$(( total_sec % 60 ))
  if (( h > 0 )); then
    printf '%dh%02dm%02ds' "$h" "$m" "$s"
  elif (( m > 0 )); then
    printf '%dm%02ds' "$m" "$s"
  else
    printf '%ds' "$s"
  fi
}

stage_stamp() {
  local stage_id="$1"
  echo "$VALIDATION_STATE_DIR/${stage_id}.done"
}

mark_stage_done() {
  local stage_id="$1"
  local started_at="$2"
  local start_epoch="$3"
  local end_epoch
  local ended_at
  local duration_sec
  local duration_human
  end_epoch="$(date +%s)"
  ended_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  duration_sec=$(( end_epoch - start_epoch ))
  duration_human="$(format_duration "$duration_sec")"
  cat >"$(stage_stamp "$stage_id")" <<EOF
done_at=$ended_at
stage_started_at=$started_at
stage_start_epoch=$start_epoch
stage_ended_at=$ended_at
stage_end_epoch=$end_epoch
duration_sec=$duration_sec
duration_human=$duration_human
EOF
}

run_stage() {
  local stage_id="$1"
  local title="$2"
  shift 2
  local stamp
  stamp="$(stage_stamp "$stage_id")"
  if [[ "$REEX_RESUME" == "1" && -f "$stamp" ]]; then
    local duration_human=""
    duration_human="$(sed -n 's/^duration_human=//p' "$stamp" | tail -n 1)"
    if [[ -n "$duration_human" ]]; then
      echo "[恢复] 跳过 ${title}（已记录耗时: ${duration_human}）"
    else
      echo "[恢复] 跳过 ${title}"
    fi
    return 0
  fi
  local started_at
  local start_epoch
  started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  start_epoch="$(date +%s)"
  if "$@"; then
    mark_stage_done "$stage_id" "$started_at" "$start_epoch"
    local duration_sec=$(( $(date +%s) - start_epoch ))
    echo "  [通过] ${title}（耗时: $(format_duration "$duration_sec")）"
    return 0
  fi
  return 1
}

prepare_build_dir() {
  local bdir="$1"
  if [[ "$VERIFY_CLEAN_BUILDS" == "1" ]]; then
    rm -rf "$bdir"
  fi
  mkdir -p "$bdir"
  if [[ "$VERIFY_CLEAN_BUILDS" == "1" ]]; then
    echo "  [build] 已清理并重建: $bdir"
  elif [[ -f "$bdir/CMakeCache.txt" ]]; then
    echo "  [build] 复用已有构建目录: $bdir"
  else
    echo "  [build] 初始化构建目录: $bdir"
  fi
}

find_ops() {
  local d="$1"
  for p in "$d/bin/test-backend-ops" "$d/tests/test-backend-ops"; do
    if [[ -x "$p" ]]; then
      OPS_BIN="$p"
      return 0
    fi
  done
  return 1
}

tier0_upstream_cpu() {
  local name="val_upstream_cpu"
  local bdir="$VERIFY_ROOT/build_${name}"
  echo ""
  echo "========== Tier 0a: 上游 CPU 基线（无 REEX CMake 选项）=========="
  echo "  目录: $bdir"
  prepare_build_dir "$bdir"
  local -a cmake_args=(
    -DCMAKE_BUILD_TYPE=Release
    -DGGML_USE_REEX=OFF
    -DGGML_REEX_FP16_PIPELINE=OFF
    -DGGML_REEX_GEMM=OFF
  )
  if [[ -n "$VERIFY_CMAKE_GENERATOR" ]]; then
    cmake_args=(-G "$VERIFY_CMAKE_GENERATOR" "${cmake_args[@]}")
  fi
  cmake -S "$VERIFY_ROOT" -B "$bdir" "${cmake_args[@]}"
  cmake --build "$bdir" -j"$VERIFY_JOBS" --target llama-perplexity test-backend-ops
  find_ops "$bdir" || { echo "[失败] 未找到 test-backend-ops"; return 1; }
  echo "  [运行] $OPS_BIN（全量 op 一致性测试，可能较慢）"
  "$OPS_BIN"
  local lp="$bdir/bin/llama-perplexity"
  if ! timeout 120 "$lp" --help >/dev/null 2>&1 \
      && ! timeout 120 "$lp" -h >/dev/null 2>&1; then
    if ! timeout 120 "$lp" -h 2>&1 | grep -qiE 'usage|perplexity|llama|common params'; then
      echo "[失败] llama-perplexity -h/--help 无预期输出"
      return 1
    fi
  fi
  echo "  [通过] Tier 0a"
}

tier0_upstream_cuda() {
  [[ "${SKIP_CUDA:-0}" == "1" ]] && { echo "[跳过] Tier 0b CUDA（SKIP_CUDA=1）"; return 0; }
  local name="val_upstream_cuda"
  local bdir="$VERIFY_ROOT/build_${name}"
  echo ""
  echo "========== Tier 0b: 上游 CUDA 基线（仅 GGML_CUDA，无 REEX）=========="
  echo "  目录: $bdir"
  prepare_build_dir "$bdir"
  local -a cmake_args=(
    -DCMAKE_BUILD_TYPE=Release
    -DGGML_CUDA=ON
    -DGGML_USE_REEX=OFF
    -DGGML_REEX_FP16_PIPELINE=OFF
    -DGGML_REEX_GEMM=OFF
  )
  if [[ -n "$VERIFY_CMAKE_GENERATOR" ]]; then
    cmake_args=(-G "$VERIFY_CMAKE_GENERATOR" "${cmake_args[@]}")
  fi
  cmake -S "$VERIFY_ROOT" -B "$bdir" "${cmake_args[@]}"
  cmake --build "$bdir" -j"$VERIFY_JOBS" --target llama-perplexity test-backend-ops
  find_ops "$bdir" || { echo "[失败] 未找到 test-backend-ops"; return 1; }
  local cuda_lib="${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}/lib64"
  local lp="$bdir/bin/llama-perplexity"
  local ldpath="$bdir/bin"
  [[ -d "$cuda_lib" ]] && ldpath="$ldpath:$cuda_lib"
  [[ -n "${LD_LIBRARY_PATH:-}" ]] && ldpath="$ldpath:$LD_LIBRARY_PATH"
  if [[ -n "${VERIFY_TIER0_CUDA_OPS:-}" ]]; then
    echo "  [运行] $OPS_BIN -o $VERIFY_TIER0_CUDA_OPS"
    env LD_LIBRARY_PATH="$ldpath" "$OPS_BIN" -o "$VERIFY_TIER0_CUDA_OPS"
  else
    echo "  [运行] $OPS_BIN"
    env LD_LIBRARY_PATH="$ldpath" "$OPS_BIN"
  fi
  if ! env LD_LIBRARY_PATH="$ldpath" timeout 120 "$lp" --help >/dev/null 2>&1 \
      && ! env LD_LIBRARY_PATH="$ldpath" timeout 120 "$lp" -h >/dev/null 2>&1; then
    if ! env LD_LIBRARY_PATH="$ldpath" timeout 120 "$lp" -h 2>&1 | grep -qiE 'usage|perplexity|llama|common params'; then
      echo "[失败] llama-perplexity -h/--help（CUDA 基线）"
      return 1
    fi
  fi
  echo "  [通过] Tier 0b"
}

tier1_reex_matrix() {
  echo ""
  echo "========== Tier 1: 自研多变体（verify_build_variants.sh）=========="
  export VERIFY_ROOT VERIFY_JOBS
  bash "$SCRIPT_DIR/verify_build_variants.sh"
}

echo "=== run_reex_full_validation.sh ==="
echo "  VERIFY_ROOT=$VERIFY_ROOT  VERIFY_JOBS=$VERIFY_JOBS"
echo "  SKIP_CUDA=${SKIP_CUDA:-0}  SKIP_TIER0=${SKIP_TIER0:-0}  SKIP_TIER1=${SKIP_TIER1:-0}"
echo "  SKIP_TIER0_CPU=${SKIP_TIER0_CPU:-0}  SKIP_TIER0_CUDA=${SKIP_TIER0_CUDA:-0}"
echo "  VERIFY_QUICK=${VERIFY_QUICK:-0}  VERIFY_GEMM_Q8=${VERIFY_GEMM_Q8:-0}"
echo "  REEX_RESUME=$REEX_RESUME  VERIFY_CLEAN_BUILDS=$VERIFY_CLEAN_BUILDS"
echo "  VERIFY_CMAKE_GENERATOR=${VERIFY_CMAKE_GENERATOR:-<cmake-default>}"
echo "  VERIFY_TIER0_CUDA_OPS=${VERIFY_TIER0_CUDA_OPS:-<full-all-ops>}"
echo "  VALIDATION_STATE_DIR=$VALIDATION_STATE_DIR"
echo "  规范文档: tests/test-reex/REEX_FULL_VALIDATION.md"
echo ""

if [[ "${SKIP_TIER0:-0}" != "1" ]]; then
  if [[ "${SKIP_TIER0_CPU:-0}" != "1" ]]; then
    run_stage "tier0a_upstream_cpu" "Tier 0a 上游 CPU 基线" tier0_upstream_cpu
  else
    echo "[跳过] Tier 0a 上游 CPU 基线（SKIP_TIER0_CPU=1）"
  fi
  if [[ "${SKIP_TIER0_CUDA:-0}" != "1" ]]; then
    run_stage "tier0b_upstream_cuda" "Tier 0b 上游 CUDA 基线" tier0_upstream_cuda
  else
    echo "[跳过] Tier 0b 上游 CUDA 基线（SKIP_TIER0_CUDA=1）"
  fi
fi

if [[ "${SKIP_TIER1:-0}" != "1" ]]; then
  run_stage "tier1_reex_matrix" "Tier 1 自研多变体矩阵" tier1_reex_matrix
fi

echo ""
echo "=== 全部完成：Tier0（上游）+ Tier1（自研矩阵）已通过 ==="
echo "  若修改 FP16/KV/算子，请再跑 Tier2:"
echo "    ./tests/test-reex/run_fp16_pipeline_precision_test.sh build_fp16 [model.gguf] [wiki.txt]"
exit 0
