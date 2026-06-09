#!/usr/bin/env bash
# 验证多种 CMake 组合能完整编译，并对二进制做最小运行时自检（--help / --list-ops）。
# 与「官网基线 + 自研矩阵」的完整规范：tests/test-reex/REEX_FULL_VALIDATION.md
# 推荐总入口：scripts/run_reex_full_validation.sh（先 Tier0 无 REEX 全量 test-backend-ops，再本脚本）
#
# 用法（在 llama.cpp 根目录）:
#   ./scripts/verify_build_variants.sh
#
# 环境变量:
#   VERIFY_ROOT          仓库根目录（默认脚本所在目录的父目录）
#   VERIFY_JOBS          并行编译线程数（默认 nproc）
#   SKIP_CUDA=1          跳过所有 GGML_CUDA=ON 的变体（无 nvcc/无 GPU 环境）
#   VERIFY_QUICK=1       仅跑最小集合：无 CUDA 时为 cpu_default + cpu_reex_fp16；有 CUDA 时再含 cuda_only + cuda_reex_fp16
#   VERIFY_GEMM_Q8=1     额外 CPU/CUDA「GEMM+Q8」与 cuda_reex_full_q8；二者 CUDA 构建均带 GGML_CUDA_FA_ALL_QUANTS=ON
#                        （混合 -ctk/-ctv 量化类型时需完整 FA 量化模板，否则易 SIGSEGV）
#   VERIFY_SKIP_RUNTIME=1  编译通过后不执行二进制自检
#   VERIFY_WIKITEXT_SMOKE=1 且设置 VERIFY_MODEL + VERIFY_WIKITEXT 时，全部通过后再跑 Wikitext 小批量 PPL（见 scripts/verify_variants_wikitext_smoke.sh）
#   CUDA_HOME / CUDA_PATH  运行时找 libcudart（默认 /usr/local/cuda）
#
# 变体说明（默认 VERIFY_QUICK=0 时）:
#   CPU: 默认 / REEX 仅 LUT / REEX+FP16 管线 / REEX GEMM / REEX GEMM+FP16（GEMM 须关 GGML_CPU_REPACK）
#   CUDA: 仅 CUDA / CUDA+REEX LUT / CUDA+REEX+FP16 / CUDA+REEX GEMM / CUDA 全量 REEX（GEMM+FP16+LUT）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERIFY_ROOT="${VERIFY_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VERIFY_JOBS="${VERIFY_JOBS:-${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc 2>/dev/null || echo 4)}}"
VERIFY_CMAKE_GENERATOR="${VERIFY_CMAKE_GENERATOR:-}"
REEX_GATE_PROFILE="${REEX_GATE_PROFILE:-full}"
VERIFY_VARIANTS="${VERIFY_VARIANTS:-}"
VERIFY_CLEAN_BUILDS="${VERIFY_CLEAN_BUILDS:-0}"
case "$REEX_GATE_PROFILE" in
  project|full) ;;
  *)
    echo "错误: 未知 REEX_GATE_PROFILE=$REEX_GATE_PROFILE（仅支持 project/full）" >&2
    exit 2
    ;;
esac
cd "$VERIFY_ROOT"

# shellcheck source=/dev/null
source "$SCRIPT_DIR/reex_cuda_env.sh"
reex_prepend_cuda_path || true

CUDA_LIB="${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}/lib64"
if [[ -z "${VERIFY_OPS_VARIANTS:-}" ]]; then
  if [[ "$REEX_GATE_PROFILE" == "project" ]]; then
    VERIFY_OPS_VARIANTS="cpu_reex_fp16 cuda_reex_fp16"
  else
    VERIFY_OPS_VARIANTS="cpu_reex_fp16 cuda_reex_fp16 cuda_reex_full_q8"
  fi
fi

PASS=0
FAIL=0
SKIP=0
# 记录完整通过（默认含运行时）的变体名，便于在冗长编译日志后核对
PASSED_NAMES=()
PROJECT_VARIANTS="${PROJECT_VARIANTS:-cpu_default cpu_reex_fp16 cuda_only cuda_reex_fp16 cuda_reex_gemm cuda_reex_full}"

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

# 运行时：能加载 .so 并打印帮助即视为通过
runtime_smoke() {
  local bdir="$1"
  local tag="$2"
  local check_ops="${3:-0}"
  local bin_dir="$bdir/bin"
  [[ "${VERIFY_SKIP_RUNTIME:-0}" == "1" ]] && return 0
  local lp="$bin_dir/llama-perplexity"
  if [[ ! -x "$lp" ]]; then
    echo "  [运行时失败] 缺少可执行文件: $lp"
    return 1
  fi
  local ldpath="$bin_dir"
  [[ -d "$CUDA_LIB" ]] && ldpath="$ldpath:$CUDA_LIB"
  [[ -n "${LD_LIBRARY_PATH:-}" ]] && ldpath="$ldpath:$LD_LIBRARY_PATH"
  if ! env LD_LIBRARY_PATH="$ldpath" timeout 60 "$lp" --help >/dev/null 2>&1 \
      && ! env LD_LIBRARY_PATH="$ldpath" timeout 60 "$lp" -h >/dev/null 2>&1; then
    # 部分构建 --help 不可用，尝试 -h
    if ! env LD_LIBRARY_PATH="$ldpath" timeout 60 "$lp" -h 2>&1 | grep -qiE 'usage|perplexity|llama'; then
      echo "  [运行时失败] llama-perplexity -h/--help 无预期输出（$tag），请检查 LD_LIBRARY_PATH / CUDA"
      return 1
    fi
  fi
  echo "  [通过] 运行时: llama-perplexity -h"
  local ops="$bin_dir/test-backend-ops"
  if [[ "$check_ops" == "1" && -x "$ops" ]]; then
    if env LD_LIBRARY_PATH="$ldpath" timeout 120 "$ops" --list-ops >/dev/null 2>&1; then
      echo "  [通过] 运行时: test-backend-ops --list-ops"
    elif env LD_LIBRARY_PATH="$ldpath" timeout 120 "$ops" --list-ops 2>&1 | grep -q .; then
      echo "  [通过] 运行时: test-backend-ops --list-ops"
    else
      echo "  [运行时失败] test-backend-ops --list-ops（$tag）"
      return 1
    fi
  fi
  return 0
}

should_build_ops_variant() {
  local name="$1"
  [[ " $VERIFY_OPS_VARIANTS " == *" $name "* ]]
}

should_run_variant() {
  local name="$1"
  if [[ -n "$VERIFY_VARIANTS" ]]; then
    [[ " $VERIFY_VARIANTS " == *" $name "* ]]
    return $?
  fi
  if [[ "$REEX_GATE_PROFILE" != "project" ]]; then
    return 0
  fi
  [[ " $PROJECT_VARIANTS " == *" $name "* ]]
}

cache_bool_is() {
  local cache="$1"
  local key="$2"
  local expect="$3"
  [[ -f "$cache" ]] || return 1
  grep -q "^${key}:BOOL=${expect}$" "$cache" 2>/dev/null
}

show_cache_bool() {
  local cache="$1"
  local key="$2"
  grep "^${key}:BOOL=" "$cache" 2>/dev/null || true
}

run_variant() {
  local name="$1"
  shift
  local -a variant_cmake_args=("$@")
  if [[ "$name" == cuda_* && -n "${VERIFY_CUDA_ARCHITECTURES:-}" ]]; then
    variant_cmake_args+=( "-DCMAKE_CUDA_ARCHITECTURES=${VERIFY_CUDA_ARCHITECTURES}" )
  fi
  if [[ -n "$VERIFY_CMAKE_GENERATOR" ]]; then
    variant_cmake_args=( -G "$VERIFY_CMAKE_GENERATOR" "${variant_cmake_args[@]}" )
  fi
  if ! should_run_variant "$name"; then
    echo ""
    echo "[project] 跳过非默认变体: $name"
    SKIP=$((SKIP + 1))
    return
  fi
  local bdir="$VERIFY_ROOT/build_verify_${name}"
  local start_epoch
  local started_at
  start_epoch="$(date +%s)"
  started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo ""
  echo "========== 变体: ${name} =========="
  echo "  目录: $bdir"
  echo "  cmake 参数: ${variant_cmake_args[*]}"
  echo "  [开始] started_at=$started_at"
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
  if ! cmake -S "$VERIFY_ROOT" -B "$bdir" \
      -DCMAKE_BUILD_TYPE=Release \
      "${variant_cmake_args[@]}"; then
    echo "  [失败] CMake 配置失败: $name"
    FAIL=$((FAIL + 1))
    return
  fi
  if [[ " ${variant_cmake_args[*]} " == *" -DGGML_CUDA_FA_ALL_QUANTS=ON "* ]]; then
    if cache_bool_is "$bdir/CMakeCache.txt" GGML_CUDA_FA_ALL_QUANTS ON; then
      echo "  [通过] CMakeCache: GGML_CUDA_FA_ALL_QUANTS=ON"
    else
      echo "  [失败] CMake 配置后未生效: GGML_CUDA_FA_ALL_QUANTS=ON"
      echo "  当前缓存:"
      show_cache_bool "$bdir/CMakeCache.txt" GGML_CUDA_FA_ALL_QUANTS
      FAIL=$((FAIL + 1))
      return
    fi
  fi
  if ! cmake --build "$bdir" -j"$VERIFY_JOBS" --target llama-perplexity; then
    echo "  [失败] 编译 llama-perplexity: $name"
    FAIL=$((FAIL + 1))
    return
  fi
  echo "  [通过] 编译: llama-perplexity"
  local check_ops=0
  if should_build_ops_variant "$name"; then
    check_ops=1
    if cmake --build "$bdir" -j"$VERIFY_JOBS" --target test-backend-ops 2>/dev/null; then
      echo "  [通过] 编译: test-backend-ops"
    else
      echo "  [失败] 编译 test-backend-ops: $name"
      FAIL=$((FAIL + 1))
      return
    fi
  else
    echo "  [跳过] test-backend-ops 编译：该变体仅做 llama-perplexity smoke"
  fi
  if ! runtime_smoke "$bdir" "$name" "$check_ops"; then
    FAIL=$((FAIL + 1))
    return
  fi
  PASS=$((PASS + 1))
  PASSED_NAMES+=("$name")
  local duration_sec=$(( $(date +%s) - start_epoch ))
  echo "  ── 变体「${name}」完成：编译 + 最小运行时自检 OK（耗时: $(format_duration "$duration_sec")）──"
}

echo "=== llama.cpp 多变体编译 + 最小运行时验证 ==="
echo "  判定: 「通过」= 该变体编译成功，且（除非 VERIFY_SKIP_RUNTIME=1）llama-perplexity 与 test-backend-ops 自检通过"
echo "  REEX_GATE_PROFILE=$REEX_GATE_PROFILE"
echo "  VERIFY_ROOT=$VERIFY_ROOT"
echo "  VERIFY_JOBS=$VERIFY_JOBS"
echo "  SKIP_CUDA=${SKIP_CUDA:-0}  VERIFY_QUICK=${VERIFY_QUICK:-0}"
echo "  VERIFY_SKIP_RUNTIME=${VERIFY_SKIP_RUNTIME:-0}"
echo "  VERIFY_CLEAN_BUILDS=$VERIFY_CLEAN_BUILDS"
echo "  VERIFY_CMAKE_GENERATOR=${VERIFY_CMAKE_GENERATOR:-<cmake-default>}"
echo "  VERIFY_OPS_VARIANTS=$VERIFY_OPS_VARIANTS"
echo "  VERIFY_VARIANTS=${VERIFY_VARIANTS:-<all-by-profile>}"
echo "  CUDA_LIB=$CUDA_LIB"
if [[ "$REEX_GATE_PROFILE" == "project" ]]; then
  echo "  PROJECT_VARIANTS=$PROJECT_VARIANTS"
fi

# ---------- CPU 变体 ----------
run_variant cpu_default

if [[ "${VERIFY_QUICK:-0}" != "1" ]]; then
  run_variant cpu_reex_lut_only \
    -DGGML_USE_REEX=ON
fi

run_variant cpu_reex_fp16 \
  -DGGML_REEX_FP16_PIPELINE=ON \
  -DGGML_USE_REEX=ON

if [[ "${VERIFY_QUICK:-0}" != "1" ]]; then
  # REEX GEMM 与 GGML_CPU_REPACK 互斥（见 ggml/CMakeLists.txt）
  run_variant cpu_reex_gemm \
    -DGGML_REEX_GEMM=ON \
    -DGGML_CPU_REPACK=OFF \
    -DGGML_USE_REEX=ON

  run_variant cpu_reex_gemm_fp16 \
    -DGGML_REEX_GEMM=ON \
    -DGGML_CPU_REPACK=OFF \
    -DGGML_USE_REEX=ON \
    -DGGML_REEX_FP16_PIPELINE=ON

  if [[ "${VERIFY_GEMM_Q8:-0}" == "1" ]]; then
    run_variant cpu_reex_gemm_q8 \
      -DGGML_REEX_GEMM=ON \
      -DGGML_REEX_GEMM_ACTIVATION=Q8 \
      -DGGML_CPU_REPACK=OFF \
      -DGGML_USE_REEX=ON
  fi
fi

# ---------- CUDA 变体 ----------
if [[ "${SKIP_CUDA:-0}" == "1" ]]; then
  echo ""
  echo "[跳过] SKIP_CUDA=1，跳过所有 cuda_* 变体"
  if [[ "${VERIFY_QUICK:-0}" == "1" ]]; then
    SKIP=$((SKIP + 2))
  else
    SKIP=$((SKIP + 5))
    [[ "${VERIFY_GEMM_Q8:-0}" == "1" ]] && SKIP=$((SKIP + 2))
  fi
else
  run_variant cuda_only -DGGML_CUDA=ON

  if [[ "${VERIFY_QUICK:-0}" != "1" ]]; then
    run_variant cuda_reex_lut \
      -DGGML_CUDA=ON \
      -DGGML_USE_REEX=ON
  fi

  run_variant cuda_reex_fp16 \
    -DGGML_CUDA=ON \
    -DGGML_REEX_FP16_PIPELINE=ON \
    -DGGML_USE_REEX=ON

  if [[ "${VERIFY_QUICK:-0}" != "1" ]]; then
    run_variant cuda_reex_gemm \
      -DGGML_CUDA=ON \
      -DGGML_REEX_GEMM=ON \
      -DGGML_CPU_REPACK=OFF \
      -DGGML_USE_REEX=ON

    run_variant cuda_reex_full \
      -DGGML_CUDA=ON \
      -DGGML_REEX_GEMM=ON \
      -DGGML_CPU_REPACK=OFF \
      -DGGML_USE_REEX=ON \
      -DGGML_REEX_FP16_PIPELINE=ON

    if [[ "${VERIFY_GEMM_Q8:-0}" == "1" ]]; then
      # FA 混合 K/V 量化（如 -ctk q8_0 -ctv q4_0）需 GGML_CUDA_FA_ALL_QUANTS，否则 fattn 选不到内核易 SIGSEGV(139)
      run_variant cuda_reex_gemm_q8 \
        -DGGML_CUDA=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DGGML_REEX_GEMM=ON \
        -DGGML_REEX_GEMM_ACTIVATION=Q8 \
        -DGGML_CPU_REPACK=OFF \
        -DGGML_USE_REEX=ON
      # GEMM(Q8) + REEX FP16 主通路 + PWNL（GGML_USE_REEX）；运行时常用 -ctk q8_0 -ctv q8_0 做 KV Q8
      run_variant cuda_reex_full_q8 \
        -DGGML_CUDA=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DGGML_REEX_GEMM=ON \
        -DGGML_REEX_GEMM_ACTIVATION=Q8 \
        -DGGML_CPU_REPACK=OFF \
        -DGGML_USE_REEX=ON \
        -DGGML_REEX_FP16_PIPELINE=ON
    fi
  fi
fi

echo ""
echo "=== 汇总 ==="
if [[ "${VERIFY_SKIP_RUNTIME:-0}" == "1" ]]; then
  echo "  通过(仅编译，VERIFY_SKIP_RUNTIME=1 已跳过二进制自检): $PASS"
else
  echo "  通过(编译 + 最小运行时): $PASS"
  echo "    说明: 每项均执行 llama-perplexity -h/--help；若编出 test-backend-ops 则再执行 --list-ops"
fi
echo "  失败: $FAIL"
echo "  跳过计数(仅提示): $SKIP"
if [[ "${#PASSED_NAMES[@]}" -gt 0 ]]; then
  echo "  已通过变体列表 (${#PASSED_NAMES[@]}):"
  for _vn in "${PASSED_NAMES[@]}"; do
    echo "    ✓ ${_vn}"
  done
fi

if [[ "$FAIL" -gt 0 ]]; then
  exit 1
fi

echo ""
echo "  [可选] Q8 激活：VERIFY_GEMM_Q8=1 可生成 build_verify_cuda_reex_gemm_q8 与 build_verify_cuda_reex_full_q8"
echo "         （后者 = GEMM Q8 + FP16 管线 + PWNL）；端到端对比: ./scripts/verify_reex_gemm_q8_e2e.sh"

if [[ "${VERIFY_WIKITEXT_SMOKE:-0}" == "1" ]]; then
  echo ""
  if [[ -n "${VERIFY_MODEL:-}" && -f "${VERIFY_MODEL}" && -n "${VERIFY_WIKITEXT:-}" && -f "${VERIFY_WIKITEXT}" ]]; then
    export VERIFY_MODEL VERIFY_WIKITEXT VERIFY_ROOT
    "$SCRIPT_DIR/verify_variants_wikitext_smoke.sh"
  else
    echo "[提示] VERIFY_WIKITEXT_SMOKE=1 但未设置有效的 VERIFY_MODEL / VERIFY_WIKITEXT 文件路径，跳过 Wikitext PPL 冒烟"
    echo "  示例: VERIFY_MODEL=/path/M.gguf VERIFY_WIKITEXT=/path/validation.txt ./scripts/verify_build_variants.sh"
  fi
fi

exit 0
