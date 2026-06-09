#!/usr/bin/env bash
# 在无 nvcc 的宿主机上，于已有 GPU/CUDA 容器内跑完整 REEX 验证。
# 依赖: Docker，容器内挂载本仓库到 REEX_VALIDATION_WORKDIR（默认 /workspace/llama.cpp）。
#
# 用法（宿主机，仓库根目录）:
#   ./scripts/run_reex_full_validation_docker.sh
#   VERIFY_QUICK=1 ./scripts/run_reex_full_validation_docker.sh
#   REEX_VALIDATION_CONTAINER=my-cuda ./scripts/run_reex_full_validation_docker.sh
#
# 与 lm_evaluator 的评测容器对齐时:
#   REEX_VALIDATION_CONTAINER=llq-eval REEX_VALIDATION_WORKDIR=/workspace/llama.cpp ./scripts/run_reex_full_validation_docker.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="${REEX_VALIDATION_CONTAINER:-llq-eval}"
WORKDIR="${REEX_VALIDATION_WORKDIR:-/workspace/llama.cpp}"

if ! docker info >/dev/null 2>&1; then
  echo "错误: 无法连接 Docker daemon。"
  exit 1
fi
if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "错误: 容器「$CONTAINER」未在运行（docker ps 中不存在）。"
  echo "  请先启动: docker start $CONTAINER"
  echo "  或新建评测容器见 lm_evaluator/scripts/run_llq_eval_container.sh"
  exit 1
fi

DOCKER_ENV=()
for n in VERIFY_ROOT VERIFY_QUICK SKIP_CUDA VERIFY_JOBS VERIFY_GEMM_Q8 VERIFY_SKIP_RUNTIME SKIP_TIER0 SKIP_TIER1; do
  if [[ -n "${!n:-}" ]]; then
    DOCKER_ENV+=( -e "$n=${!n}" )
  fi
done

echo "=== run_reex_full_validation_docker.sh ==="
echo "  容器: $CONTAINER"
echo "  工作目录: $WORKDIR"
echo "  在容器内执行: ./scripts/run_reex_full_validation.sh"
echo ""

docker exec \
  "${DOCKER_ENV[@]}" \
  "$CONTAINER" \
  bash -lc "set -euo pipefail; cd '$WORKDIR' && exec ./scripts/run_reex_full_validation.sh"
