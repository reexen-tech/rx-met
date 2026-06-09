#!/usr/bin/env bash
# REEX 统一验证入口。
# 用法:
#   ./scripts/reex_validate.sh help
#   ./scripts/reex_validate.sh baseline
#   ./scripts/reex_validate.sh matrix
#   ./scripts/reex_validate.sh baseline-docker
#   ./scripts/reex_validate.sh matrix-docker
#   ./scripts/reex_validate.sh fp16 [build_dir] [model.gguf] [wiki.txt]
#   ./scripts/reex_validate.sh cuda-ab
#   ./scripts/reex_validate.sh q8-e2e
#   ./scripts/reex_validate.sh gate-project [build_dir] [q4_model.gguf] [wiki.txt]
#   ./scripts/reex_validate.sh gate-project-docker [build_dir] [q4_model.gguf] [wiki.txt]
#   ./scripts/reex_validate.sh gate-full [build_dir] [q4_model.gguf] [wiki.txt]
#   ./scripts/reex_validate.sh gate-full-docker [build_dir] [q4_model.gguf] [wiki.txt]
#   ./scripts/reex_validate.sh gate-all [build_dir] [q4_model.gguf] [wiki.txt]
#   ./scripts/reex_validate.sh gate-all-docker [build_dir] [q4_model.gguf] [wiki.txt]
#   ./scripts/reex_validate.sh all
#
# 设计目标:
# - 对日常使用者暴露尽量少的命令
# - 保留现有脚本为底层实现，避免重复维护逻辑
# - 明确“改什么代码，就跑什么测试”
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

usage() {
  cat <<'EOF'
REEX 验证统一入口

子命令:
  help
      显示帮助。

  baseline
      跑 Tier0 + Tier1。
      等价于: ./scripts/run_reex_full_validation.sh

  matrix
      只跑 Tier1 多变体矩阵。
      等价于: SKIP_TIER0=1 ./scripts/run_reex_full_validation.sh

  baseline-docker
      在 Docker/CUDA 容器中跑 Tier0 + Tier1。
      等价于: ./scripts/run_reex_full_validation_docker.sh

  matrix-docker
      在 Docker/CUDA 容器中跑 Tier1 多变体矩阵。
      等价于: SKIP_TIER0=1 ./scripts/run_reex_full_validation_docker.sh

  fp16 [build_dir] [model.gguf] [wiki.txt]
      跑 FP16 管线 backend-ops；若给模型和数据，再加 PPL。
      等价于: ./tests/test-reex/run_fp16_pipeline_precision_test.sh ...

  cuda-ab
      跑 CUDA A/B、batch、Q16 trace 等专项验证。
      等价于: ./scripts/verify_build_cuda_reex.sh

  q8-e2e
      跑 GEMM Q8 端到端验证。
      等价于: ./scripts/verify_reex_gemm_q8_e2e.sh

  gate-project [build_dir] [q4_model.gguf] [wiki.txt]
      默认项目门禁，面向日常开发与快速上库验证。
      等价于: REEX_GATE_PROFILE=project ./scripts/run_reex_all_function_tests.sh

  gate-project-docker [build_dir] [q4_model.gguf] [wiki.txt]
      在 Docker/CUDA 容器中执行默认项目门禁。
      等价于: REEX_GATE_PROFILE=project ./scripts/run_reex_all_function_tests_docker.sh

  gate-full [build_dir] [q4_model.gguf] [wiki.txt]
      完整深度回归门禁，保留当前所有功能测试类别。
      等价于: REEX_GATE_PROFILE=full ./scripts/run_reex_all_function_tests.sh

  gate-full-docker [build_dir] [q4_model.gguf] [wiki.txt]
      在 Docker/CUDA 容器中执行完整深度回归门禁。
      等价于: REEX_GATE_PROFILE=full ./scripts/run_reex_all_function_tests_docker.sh

  gate-all [build_dir] [q4_model.gguf] [wiki.txt]
      兼容旧入口；默认等价于 gate-project。
      project 档包含：
      1) baseline
      2) test-reex-lut
      3) test-reex-gemm
      4) fp16
      5) verify_variants_wikitext_smoke（若给 q4_model + wiki）
      6) cuda-ab（无 SKIP_CUDA 时）
      full 档再额外包含：
      7) q8-e2e（无 SKIP_CUDA 且给 q4_model + wiki）
      8) run_single_block_compare（若额外给 MODEL_F16 + MODEL_Q4_K）

  gate-all-docker [build_dir] [q4_model.gguf] [wiki.txt]
      兼容旧入口；默认等价于 gate-project-docker。

  all [build_dir] [model.gguf] [wiki.txt]
      建议的完整顺序:
      1) baseline
      2) fp16
      3) 若改了 CUDA，再跑 cuda-ab
      4) 若改了 GEMM Q8，再跑 q8-e2e

常用环境变量:
  SKIP_CUDA=1
  VERIFY_QUICK=1
  VERIFY_GEMM_Q8=1
  VERIFY_SKIP_RUNTIME=1
  VERIFY_JOBS=<n>
  REEX_VALIDATION_CONTAINER=llama.cpp-ci-docker
  REEX_GATE_PROFILE=project|full

改动 -> 推荐命令:
  只改文档/脚本:
    ./scripts/reex_validate.sh baseline

  改 llama 层图构建 / KV 边界 / 模型接线:
    ./scripts/reex_validate.sh baseline
    ./scripts/reex_validate.sh fp16 build_fp16

  改 FP16 pipeline / ggml op / CPU 路径:
    ./scripts/reex_validate.sh baseline
    ./scripts/reex_validate.sh fp16 build_fp16 [model.gguf] [wiki.txt]

  改 CUDA kernel / 调度 / Flash-Attn:
    ./scripts/reex_validate.sh baseline
    ./scripts/reex_validate.sh fp16 build_fp16 [model.gguf] [wiki.txt]
    ./scripts/reex_validate.sh cuda-ab

  改 GEMM Q8 / KV Q8-Q4 组合:
    VERIFY_GEMM_Q8=1 ./scripts/reex_validate.sh baseline
    ./scripts/reex_validate.sh q8-e2e

更多说明:
  scripts/REEX_VALIDATION_GUIDE.md
  tests/test-reex/REEX_FULL_VALIDATION.md
EOF
}

cmd="${1:-help}"
shift || true

cd "$REPO_ROOT"

case "$cmd" in
  help|-h|--help)
    usage
    ;;
  baseline)
    exec "$SCRIPT_DIR/run_reex_full_validation.sh" "$@"
    ;;
  matrix)
    exec env SKIP_TIER0=1 "${SCRIPT_DIR}/run_reex_full_validation.sh" "$@"
    ;;
  baseline-docker)
    exec "${SCRIPT_DIR}/run_reex_full_validation_docker.sh" "$@"
    ;;
  matrix-docker)
    exec env SKIP_TIER0=1 "${SCRIPT_DIR}/run_reex_full_validation_docker.sh" "$@"
    ;;
  fp16)
    exec "$REPO_ROOT/tests/test-reex/run_fp16_pipeline_precision_test.sh" "$@"
    ;;
  cuda-ab)
    exec "$SCRIPT_DIR/verify_build_cuda_reex.sh" "$@"
    ;;
  q8-e2e)
    exec "$SCRIPT_DIR/verify_reex_gemm_q8_e2e.sh" "$@"
    ;;
  gate-project)
    exec env REEX_GATE_PROFILE=project "$SCRIPT_DIR/run_reex_all_function_tests.sh" "$@"
    ;;
  gate-project-docker)
    exec env REEX_GATE_PROFILE=project "$SCRIPT_DIR/run_reex_all_function_tests_docker.sh" "$@"
    ;;
  gate-full)
    exec env REEX_GATE_PROFILE=full "$SCRIPT_DIR/run_reex_all_function_tests.sh" "$@"
    ;;
  gate-full-docker)
    exec env REEX_GATE_PROFILE=full "$SCRIPT_DIR/run_reex_all_function_tests_docker.sh" "$@"
    ;;
  gate-all)
    exec env REEX_GATE_PROFILE="${REEX_GATE_PROFILE:-project}" "$SCRIPT_DIR/run_reex_all_function_tests.sh" "$@"
    ;;
  gate-all-docker)
    exec env REEX_GATE_PROFILE="${REEX_GATE_PROFILE:-project}" "$SCRIPT_DIR/run_reex_all_function_tests_docker.sh" "$@"
    ;;
  all)
    "$SCRIPT_DIR/run_reex_full_validation.sh"
    "$REPO_ROOT/tests/test-reex/run_fp16_pipeline_precision_test.sh" "$@"
    echo ""
    echo "[提示] 若这次改动涉及 CUDA kernel / 调度，请继续执行:"
    echo "  ./scripts/reex_validate.sh cuda-ab"
    echo "[提示] 若这次改动涉及 GEMM Q8 / KV Q8-Q4，请继续执行:"
    echo "  VERIFY_GEMM_Q8=1 ./scripts/reex_validate.sh q8-e2e"
    ;;
  *)
    echo "错误: 未知子命令: $cmd" >&2
    echo "" >&2
    usage >&2
    exit 2
    ;;
esac
