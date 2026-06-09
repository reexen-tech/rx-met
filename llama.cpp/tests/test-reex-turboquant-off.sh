#!/usr/bin/env bash
# tests/test-reex-turboquant-off.sh
#
# REEX_TURBOQUANT=OFF 兼容回归脚本。
#
# 目标：
#   验证当本地 fork 关闭 TurboQuant 集成开关（即 cmake -DREEX_TURBOQUANT=OFF）
#   时，整个仓库的构建/测试行为与 upstream llama.cpp **位级等价**或至少
#   功能等价：
#
#     1. 顶层与 ggml 子项目都能 cmake configure + build；
#     2. -DREEX_TURBOQUANT 不会出现在任何编译单元的命令行中；
#     3. tests/oracle/、src/reex/turboquant_*、ggml/src/ggml-*/reex/turboquant_*
#        都不会被加入构建（避免不小心污染 OFF 路径）；
#     4. 既有的回归测试（test-quantize-fns / test-backend-ops / test-rope ...）
#        全部通过；
#     5. ggml.h 中 GGML_TYPE_TQ_* / GGML_OP_REEX_WHT 等新增枚举项在
#        REEX_TURBOQUANT=OFF 时不出现在 enum 体内（grep 兜底）。
#
# 这是 docs/turboquant/01_设计与实现计划.md §4.3 同步 SOP 的核心保护：每次合并
# 上游 main 之后，必须保证 OFF 档没有任何 TurboQuant 痕迹。
#
# 用法：
#   tests/test-reex-turboquant-off.sh            # 默认在 build-tq-off/ 里 configure + 全测
#   BUILD_DIR=build-foo tests/test-reex-turboquant-off.sh
#   tests/test-reex-turboquant-off.sh --configure-only   # 只 cmake configure
#   tests/test-reex-turboquant-off.sh --build-only       # 跳过 ctest
#
# 退出码：
#   0 = 全部通过
#   1 = 某项检查失败（脚本会打印第一条失败原因后立即退出）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build-tq-off}"
JOBS="${JOBS:-2}"
DO_CONFIGURE=1
DO_BUILD=1
DO_CTEST=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --configure-only) DO_BUILD=0; DO_CTEST=0; shift ;;
    --build-only)     DO_CTEST=0; shift ;;
    --no-ctest)       DO_CTEST=0; shift ;;
    --build-dir)      BUILD_DIR="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,40p' "$0"
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

cd "$REPO_ROOT"

echo "[off-regress] REPO  = $REPO_ROOT"
echo "[off-regress] BUILD = $BUILD_DIR"
echo "[off-regress] JOBS  = $JOBS"

# ───────────────────────────────────────────── step 1: configure (REEX_TURBOQUANT=OFF)
if [[ $DO_CONFIGURE -eq 1 ]]; then
  rm -rf "$BUILD_DIR"
  # NOTE: We deliberately keep LLAMA_BUILD_TOOLS=ON here even though we don't
  # exercise any tool binaries.  Upstream `tests/CMakeLists.txt` unconditionally
  # builds `test-mtmd-c-api`, which links against the `mtmd` target — and that
  # target is only configured when LLAMA_BUILD_TOOLS=ON.  If we turn tools off
  # the OFF-regression itself fails to compile, even though the failure is
  # entirely unrelated to REEX_TURBOQUANT.  Keeping tools=ON sidesteps the
  # issue without touching upstream files.
  cmake -S . -B "$BUILD_DIR" \
      -DREEX_TURBOQUANT=OFF \
      -DLLAMA_BUILD_TESTS=ON \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_TOOLS=ON \
      -DLLAMA_CURL=OFF \
      -DGGML_OPENMP=OFF
fi

# ───────────────────────────────────────────── step 2: assert no -DREEX_TURBOQUANT
echo "[off-regress] check 1/3 : -DREEX_TURBOQUANT must NOT appear in compile_commands.json"
if [[ -f "$BUILD_DIR/compile_commands.json" ]]; then
  if grep -q -- '-DREEX_TURBOQUANT' "$BUILD_DIR/compile_commands.json"; then
    echo "[off-regress] FAIL: -DREEX_TURBOQUANT leaked into a compile command (see below)"
    grep -n -- '-DREEX_TURBOQUANT' "$BUILD_DIR/compile_commands.json" | head -5
    exit 1
  fi
fi

# ───────────────────────────────────────────── step 3: assert reex turboquant target not built
echo "[off-regress] check 2/3 : reex_turboquant_oracle target must NOT exist"
if [[ -d "$BUILD_DIR/tests/oracle" ]]; then
  echo "[off-regress] FAIL: $BUILD_DIR/tests/oracle exists; oracle should not be configured when REEX_TURBOQUANT=OFF"
  exit 1
fi

echo "[off-regress] check 3/3 : reex_turboquant_*.{c,cu,cpp,h,cuh} must NOT appear in compile_commands.json"
if [[ -f "$BUILD_DIR/compile_commands.json" ]]; then
  if grep -E -q 'reex_turboquant_' "$BUILD_DIR/compile_commands.json"; then
    echo "[off-regress] FAIL: a reex_turboquant_* source slipped into the build:"
    grep -nE 'reex_turboquant_' "$BUILD_DIR/compile_commands.json" | head -5
    exit 1
  fi
fi

# ───────────────────────────────────────────── step 4: build
if [[ $DO_BUILD -eq 1 ]]; then
  echo "[off-regress] cmake --build (jobs=$JOBS)"
  cmake --build "$BUILD_DIR" -j "$JOBS"
fi

# ───────────────────────────────────────────── step 5: ctest (regression, no labels)
#
# We deliberately exclude tests that depend on downloading model fixtures from
# the network (`test-download-model` and anything declaring it as a CTest
# dependency).  Those are upstream tests, fail in any offline / firewalled
# environment, and have nothing to do with REEX_TURBOQUANT.  Override with
# OFF_CTEST_EXCLUDE='' if you want to surface those failures explicitly.
OFF_CTEST_EXCLUDE="${OFF_CTEST_EXCLUDE:-^test-download-model$|^test-thread-safety$|^test-state-restore-fragmented$}"
if [[ $DO_CTEST -eq 1 ]]; then
  if [[ -n "$OFF_CTEST_EXCLUDE" ]]; then
    echo "[off-regress] ctest (label = main, excluding: $OFF_CTEST_EXCLUDE)"
    ( cd "$BUILD_DIR" && ctest --output-on-failure -L main -E "$OFF_CTEST_EXCLUDE" -j "$JOBS" )
  else
    echo "[off-regress] ctest (label = main, no exclusions)"
    ( cd "$BUILD_DIR" && ctest --output-on-failure -L main -j "$JOBS" )
  fi
fi

# ───────────────────────────────────────────── step 6: header sanity (best-effort grep)
echo "[off-regress] header sanity check: ggml.h"
HEADER="$REPO_ROOT/ggml/include/ggml.h"
if [[ -f "$HEADER" ]]; then
  # Check the 10 lines preceding each TurboQuant identifier for a
  # `#ifdef REEX_TURBOQUANT` (or BEGIN comment) guard.  10 lines is roughly the
  # block-comment + ifdef header we use everywhere; tweak if the layout grows.
  if grep -q '^[[:space:]]*GGML_TYPE_TQ_K3' "$HEADER"; then
    if ! grep -B10 'GGML_TYPE_TQ_K3' "$HEADER" | grep -q 'REEX_TURBOQUANT'; then
      echo "[off-regress] WARN: GGML_TYPE_TQ_K3 found without REEX_TURBOQUANT guard nearby; review manually"
    fi
  fi
  if grep -q '^[[:space:]]*GGML_OP_REEX_WHT' "$HEADER"; then
    if ! grep -B10 'GGML_OP_REEX_WHT' "$HEADER" | grep -q 'REEX_TURBOQUANT'; then
      echo "[off-regress] WARN: GGML_OP_REEX_WHT found without REEX_TURBOQUANT guard nearby; review manually"
    fi
  fi
fi

echo "[off-regress] PASS"
