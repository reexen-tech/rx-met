#!/usr/bin/env bash
# REEX 验证总入口：支持两种档位。
#   - project: 默认项目门禁，强调日常可用时长
#   - full:    完整深度回归，尽可能覆盖所有功能测试类别
# full 档默认包含：
#   1) Tier0 + Tier1（上游基线 + 自研矩阵）
#   2) REEX LUT 专项（test-reex-lut）
#   3) REEX GEMM CPU 专项（test-reex-gemm）
#   4) FP16 pipeline backend-ops；若给 model/wiki，再加 FP16 PPL
#   5) 各 build_verify_* 的 Wikitext PPL 冒烟（需 model/wiki）
#   6) CUDA A/B、batch、Q16 trace（无 SKIP_CUDA 时）
#   7) GEMM Q8 / KV Q8-Q4 端到端（无 SKIP_CUDA 且有 model/wiki 时）
#   8) single block/full-layer 对比（需 MODEL_F16 + MODEL_Q4_K）
#
# 用法：
#   ./scripts/run_reex_all_function_tests.sh
#   REEX_GATE_PROFILE=full ./scripts/run_reex_all_function_tests.sh
#   ./scripts/run_reex_all_function_tests.sh build_fp16 /path/q4.gguf /path/wiki.txt
#   MODEL_F16=/path/f16.gguf MODEL_Q4_K=/path/q4.gguf ./scripts/run_reex_all_function_tests.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
FP16_BUILD_DIR="${1:-build_fp16_gate}"
MODEL_Q4_ARG="${2:-${VERIFY_MODEL:-${MODEL:-}}}"
WIKI_ARG="${3:-${VERIFY_WIKITEXT:-${WIKITEXT_FILE:-${DATA:-}}}}"

cd "$REPO_ROOT"

# shellcheck source=/dev/null
source "$SCRIPT_DIR/reex_cuda_env.sh"
reex_prepend_cuda_path || true

REEX_GATE_PROFILE="${REEX_GATE_PROFILE:-project}"
case "$REEX_GATE_PROFILE" in
  project|full) ;;
  *)
    echo "错误: 未知 REEX_GATE_PROFILE=$REEX_GATE_PROFILE（仅支持 project/full）" >&2
    exit 1
    ;;
esac
export REEX_GATE_PROFILE

if [[ "$REEX_GATE_PROFILE" == "project" ]]; then
  export VERIFY_QUICK="${VERIFY_QUICK:-0}"
  export VERIFY_GEMM_Q8="${VERIFY_GEMM_Q8:-0}"
else
  if [[ "${VERIFY_QUICK:-0}" == "1" ]]; then
    echo "[提示] full 档默认需要完整矩阵，忽略 VERIFY_QUICK=1，改为 VERIFY_QUICK=0"
  fi
  export VERIFY_QUICK=0
  export VERIFY_GEMM_Q8="${VERIFY_GEMM_Q8:-1}"
fi

MODEL_Q4_K="${MODEL_Q4_K:-$MODEL_Q4_ARG}"
MODEL_F16="${MODEL_F16:-}"
WIKI_FILE="${WIKI_ARG:-}"
REEX_RESUME="${REEX_RESUME:-1}"
VERIFY_CMAKE_GENERATOR="${VERIFY_CMAKE_GENERATOR:-}"
STATE_DIR="${REEX_GATE_ALL_STATE_DIR:-$REPO_ROOT/.reex-validation/gate-${REEX_GATE_PROFILE}}"

if [[ "$REEX_RESUME" != "1" ]]; then
  rm -rf "$STATE_DIR"
fi
mkdir -p "$STATE_DIR"

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

GATE_META_FILE="$STATE_DIR/gate.meta"
if [[ -f "$GATE_META_FILE" ]]; then
  # shellcheck source=/dev/null
  source "$GATE_META_FILE"
else
  gate_profile="$REEX_GATE_PROFILE"
  gate_started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  gate_start_epoch="$(date +%s)"
  cat >"$GATE_META_FILE" <<EOF
gate_profile=$gate_profile
gate_started_at=$gate_started_at
gate_start_epoch=$gate_start_epoch
EOF
fi

step_stamp() {
  local step_id="$1"
  echo "$STATE_DIR/${step_id}.done"
}

mark_step_done() {
  local step_id="$1"
  local start_epoch="$2"
  local started_at="$3"
  local end_epoch
  local ended_at
  local duration_sec
  local duration_human
  end_epoch="$(date +%s)"
  ended_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  duration_sec=$(( end_epoch - start_epoch ))
  duration_human="$(format_duration "$duration_sec")"
  cat >"$(step_stamp "$step_id")" <<EOF
done_at=$ended_at
step_started_at=$started_at
step_start_epoch=$start_epoch
step_ended_at=$ended_at
step_end_epoch=$end_epoch
duration_sec=$duration_sec
duration_human=$duration_human
EOF
  rm -f "$STATE_DIR/last_failed_step"
}

mark_step_failed() {
  local step_id="$1"
  printf '%s\n' "$step_id" >"$STATE_DIR/last_failed_step"
}

find_bin() {
  local build_dir="$1"
  local name="$2"
  for p in "$build_dir/bin/$name" "$build_dir/tests/$name" "$build_dir/$name"; do
    if [[ -x "$p" ]]; then
      echo "$p"
      return 0
    fi
  done
  return 1
}

have_gate_model_inputs() {
  [[ -n "${MODEL_Q4_K:-}" && -f "${MODEL_Q4_K:-}" && -n "${WIKI_FILE:-}" && -f "${WIKI_FILE:-}" ]]
}

run_build_target() {
  local build_dir="$1"
  local target="$2"
  cmake --build "$build_dir" -j"${VERIFY_JOBS:-${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc 2>/dev/null || echo 4)}}" --target "$target"
}

configure_build_dir() {
  local build_dir="$1"
  shift
  mkdir -p "$build_dir"
  local -a cmake_args=(
    -DCMAKE_BUILD_TYPE=Release
    "$@"
  )
  if [[ -n "$VERIFY_CMAKE_GENERATOR" ]]; then
    cmake_args=( -G "$VERIFY_CMAKE_GENERATOR" "${cmake_args[@]}" )
  fi
  cmake -S "$REPO_ROOT" -B "$build_dir" "${cmake_args[@]}"
}

run_binary_in_build() {
  local build_dir="$1"
  local name="$2"
  local ldpath="$build_dir/bin"
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    ldpath="$ldpath:$LD_LIBRARY_PATH"
  fi
  local bin
  bin="$(find_bin "$build_dir" "$name")" || {
    echo "[失败] 未找到可执行文件: $name (build=$build_dir)" >&2
    return 1
  }
  env LD_LIBRARY_PATH="$ldpath" "$bin"
}

run_step() {
  local step_id="$1"
  local title="$2"
  shift
  shift
  local stamp
  stamp="$(step_stamp "$step_id")"
  if [[ -f "$stamp" ]]; then
    local duration_human=""
    duration_human="$(sed -n 's/^duration_human=//p' "$stamp" | tail -n 1)"
    echo ""
    if [[ -n "$duration_human" ]]; then
      echo "[恢复] 跳过已完成步骤: $title（已记录耗时: $duration_human）"
    else
      echo "[恢复] 跳过已完成步骤: $title"
    fi
    return 0
  fi
  local step_start_epoch
  local step_started_at
  local duration_sec
  local duration_human
  step_start_epoch="$(date +%s)"
  step_started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo ""
  echo "================================================================"
  echo "  $title"
  echo "================================================================"
  echo "  [开始] started_at=$step_started_at"
  if "$@"; then
    mark_step_done "$step_id" "$step_start_epoch" "$step_started_at"
    duration_sec=$(( $(date +%s) - step_start_epoch ))
    duration_human="$(format_duration "$duration_sec")"
    echo "[通过] $title（耗时: $duration_human）"
    return 0
  fi
  mark_step_failed "$step_id"
  return 1
}

run_lut_suite() {
  configure_build_dir "$REPO_ROOT/build_verify_cpu_reex_lut_only" \
    -DGGML_USE_REEX=ON
  run_build_target "$REPO_ROOT/build_verify_cpu_reex_lut_only" test-reex-lut
  run_binary_in_build "$REPO_ROOT/build_verify_cpu_reex_lut_only" test-reex-lut
}

run_gemm_cpu_suite() {
  configure_build_dir "$REPO_ROOT/build_verify_cpu_reex_gemm" \
    -DGGML_REEX_GEMM=ON \
    -DGGML_CPU_REPACK=OFF \
    -DGGML_USE_REEX=ON
  run_build_target "$REPO_ROOT/build_verify_cpu_reex_gemm" test-reex-gemm
  run_binary_in_build "$REPO_ROOT/build_verify_cpu_reex_gemm" test-reex-gemm
}

echo "=== run_reex_all_function_tests.sh ==="
echo "  REEX_GATE_PROFILE=$REEX_GATE_PROFILE"
echo "  REPO_ROOT=$REPO_ROOT"
echo "  FP16_BUILD_DIR=$FP16_BUILD_DIR"
echo "  MODEL_Q4_K=${MODEL_Q4_K:-<未提供>}"
echo "  MODEL_F16=${MODEL_F16:-<未提供>}"
echo "  WIKI=${WIKI_FILE:-<未提供>}"
echo "  SKIP_CUDA=${SKIP_CUDA:-0}"
echo "  VERIFY_GEMM_Q8=$VERIFY_GEMM_Q8"
echo "  VERIFY_CMAKE_GENERATOR=${VERIFY_CMAKE_GENERATOR:-<cmake-default>}"
echo "  REEX_RESUME=$REEX_RESUME"
echo "  STATE_DIR=$STATE_DIR"
echo "  GATE_STARTED_AT=$gate_started_at"
echo ""
echo "将覆盖的功能测试："
if [[ "$REEX_GATE_PROFILE" == "project" ]]; then
  echo "  1) Project build matrix（verify_build_variants.sh 收缩矩阵）"
  echo "  2) test-reex-lut"
  echo "  3) test-reex-gemm"
  echo "  4) FP16 pipeline backend-ops + 可选 PPL（缩小 smoke 参数）"
  echo "  5) verify_variants_wikitext_smoke.sh（默认仅关键 GPU 变体）"
  echo "  6) verify_build_cuda_reex.sh（仅核心 CUDA A/B）"
  echo "  [full-only] 7) verify_reex_gemm_q8_e2e.sh"
  echo "  [full-only] 8) run_single_block_compare.sh"

  run_step "01_project_matrix" "1/6 Project build matrix" \
    env REEX_GATE_PROFILE="$REEX_GATE_PROFILE" SKIP_TIER0=1 \
    "$SCRIPT_DIR/run_reex_full_validation.sh"

  run_step "02_reex_lut" "2/6 REEX LUT 精度专项 (test-reex-lut)" run_lut_suite

  run_step "03_reex_gemm_cpu" "3/6 REEX GEMM CPU 专项 (test-reex-gemm)" run_gemm_cpu_suite

  run_step "04_fp16_pipeline" "4/6 FP16 pipeline 专项" \
    env REEX_GATE_PROFILE="$REEX_GATE_PROFILE" \
    bash "$REPO_ROOT/tests/test-reex/run_fp16_pipeline_precision_test.sh" \
    "$FP16_BUILD_DIR" "${MODEL_Q4_K:-}" "${WIKI_FILE:-}"

  if have_gate_model_inputs; then
    run_step "05_wikitext_smoke" "5/6 各 build_verify_* 的 Wikitext PPL 冒烟" \
      env REEX_GATE_PROFILE="$REEX_GATE_PROFILE" VERIFY_MODEL="$MODEL_Q4_K" VERIFY_WIKITEXT="$WIKI_FILE" \
      "$SCRIPT_DIR/verify_variants_wikitext_smoke.sh"
  else
    echo ""
    echo "[跳过] 5/6 Wikitext PPL 冒烟：未提供 Q4 模型或 wiki 文本"
  fi

  if [[ "${SKIP_CUDA:-0}" != "1" ]] && have_gate_model_inputs; then
    run_step "06_cuda_ab" "6/6 CUDA A/B 核心专项" \
      env REEX_GATE_PROFILE="$REEX_GATE_PROFILE" MODEL="${MODEL_Q4_K:-${MODEL:-}}" WIKI="${WIKI_FILE:-${WIKI:-}}" \
      "$SCRIPT_DIR/verify_build_cuda_reex.sh"
  else
    echo ""
    if [[ "${SKIP_CUDA:-0}" == "1" ]]; then
      echo "[跳过] 6/6 CUDA A/B：SKIP_CUDA=1"
    else
      echo "[跳过] 6/6 CUDA A/B：未提供可用的模型与 wiki 文本"
    fi
  fi

  echo ""
  echo "[full-only] 跳过 Q8 E2E 与 single-block/full-layer 深度回归；如需完整覆盖，请使用 REEX_GATE_PROFILE=full。"
else
  echo "  1) Tier0 + Tier1（run_reex_full_validation.sh）"
  echo "  2) test-reex-lut"
  echo "  3) test-reex-gemm"
  echo "  4) FP16 pipeline backend-ops + 可选 PPL"
  echo "  5) verify_variants_wikitext_smoke.sh（需 Q4 模型 + wiki）"
  echo "  6) verify_build_cuda_reex.sh（无 SKIP_CUDA 时）"
  echo "  7) verify_reex_gemm_q8_e2e.sh（无 SKIP_CUDA 且有 Q4 模型 + wiki）"
  echo "  8) run_single_block_compare.sh（需 F16 模型 + Q4 模型）"

  run_step "01_tier0_tier1" "1/8 Tier0 + Tier1" \
    env REEX_GATE_PROFILE="$REEX_GATE_PROFILE" \
    "$SCRIPT_DIR/run_reex_full_validation.sh"

  run_step "02_reex_lut" "2/8 REEX LUT 精度专项 (test-reex-lut)" run_lut_suite

  run_step "03_reex_gemm_cpu" "3/8 REEX GEMM CPU 专项 (test-reex-gemm)" run_gemm_cpu_suite

  run_step "04_fp16_pipeline" "4/8 FP16 pipeline 专项" \
    env REEX_GATE_PROFILE="$REEX_GATE_PROFILE" \
    bash "$REPO_ROOT/tests/test-reex/run_fp16_pipeline_precision_test.sh" \
    "$FP16_BUILD_DIR" "${MODEL_Q4_K:-}" "${WIKI_FILE:-}"

  if have_gate_model_inputs; then
    run_step "05_wikitext_smoke" "5/8 各 build_verify_* 的 Wikitext PPL 冒烟" \
      env REEX_GATE_PROFILE="$REEX_GATE_PROFILE" VERIFY_MODEL="$MODEL_Q4_K" VERIFY_WIKITEXT="$WIKI_FILE" \
      "$SCRIPT_DIR/verify_variants_wikitext_smoke.sh"
  else
    echo ""
    echo "[跳过] 5/8 Wikitext PPL 冒烟：未提供 Q4 模型或 wiki 文本"
  fi

  if [[ "${SKIP_CUDA:-0}" != "1" ]] && have_gate_model_inputs; then
    run_step "06_cuda_ab" "6/8 CUDA A/B + batch + Q16 trace" \
      env REEX_GATE_PROFILE="$REEX_GATE_PROFILE" MODEL="${MODEL_Q4_K:-${MODEL:-}}" WIKI="${WIKI_FILE:-${WIKI:-}}" \
      "$SCRIPT_DIR/verify_build_cuda_reex.sh"
  else
    echo ""
    if [[ "${SKIP_CUDA:-0}" == "1" ]]; then
      echo "[跳过] 6/8 CUDA A/B：SKIP_CUDA=1"
    else
      echo "[跳过] 6/8 CUDA A/B：未提供可用的模型与 wiki 文本"
    fi
  fi

  if [[ "${SKIP_CUDA:-0}" != "1" ]] && have_gate_model_inputs; then
    run_step "07_gemm_q8_e2e" "7/8 GEMM Q8 / KV Q8-Q4 端到端" \
      env REEX_GATE_PROFILE="$REEX_GATE_PROFILE" MODEL="$MODEL_Q4_K" WIKI="$WIKI_FILE" \
      "$SCRIPT_DIR/verify_reex_gemm_q8_e2e.sh"
  else
    echo ""
    echo "[跳过] 7/8 GEMM Q8 E2E：需要 CUDA 且提供可用的 Q4 模型 + wiki"
  fi

  if [[ -n "${MODEL_F16:-}" && -n "${MODEL_Q4_K:-}" ]]; then
    run_step "08_single_block_compare" "8/8 full-layer/single-block 对比" \
      env REEX_GATE_PROFILE="$REEX_GATE_PROFILE" MODEL_F16="$MODEL_F16" MODEL_Q4_K="$MODEL_Q4_K" \
      "$REPO_ROOT/tests/test-reex/run_single_block_compare.sh"
  else
    echo ""
    echo "[跳过] 8/8 single-block/full-layer 对比：需要同时提供 MODEL_F16 与 MODEL_Q4_K"
  fi
fi

echo ""
gate_end_epoch="$(date +%s)"
gate_ended_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
gate_duration_sec=$(( gate_end_epoch - gate_start_epoch ))
gate_duration_human="$(format_duration "$gate_duration_sec")"
cat >"$GATE_META_FILE" <<EOF
gate_profile=$REEX_GATE_PROFILE
gate_started_at=$gate_started_at
gate_start_epoch=$gate_start_epoch
gate_ended_at=$gate_ended_at
gate_end_epoch=$gate_end_epoch
gate_duration_sec=$gate_duration_sec
gate_duration_human=$gate_duration_human
EOF
echo "=== REEX ${REEX_GATE_PROFILE} gate 完成 ==="
echo "  GATE_ENDED_AT=$gate_ended_at"
echo "  GATE_TOTAL_DURATION=$gate_duration_human"
