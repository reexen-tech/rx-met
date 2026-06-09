#!/usr/bin/env bash
# 在 Docker/CUDA 容器中执行 REEX 全功能门禁。
# 默认容器: llama.cpp-ci-docker
# 默认工作目录: /workspace/llama.cpp
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="${REEX_VALIDATION_CONTAINER:-llama.cpp-ci-docker}"
WORKDIR="${REEX_VALIDATION_WORKDIR:-/workspace/llama.cpp}"

if ! docker info >/dev/null 2>&1; then
  echo "错误: 无法连接 Docker daemon。"
  exit 1
fi
if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "错误: 容器「$CONTAINER」未在运行。"
  echo "  请先执行: docker start $CONTAINER"
  exit 1
fi

if [[ "${SKIP_CUDA:-0}" != "1" ]]; then
  if ! docker exec "$CONTAINER" bash -lc 'command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1'; then
    echo "错误: 容器「$CONTAINER」当前无法访问 GPU。"
    echo "  已检测到这是一次需要 CUDA 的验证，但容器内的 nvidia-smi 不可用。"
    echo "  常见原因: 宿主机驱动重载后，长时间运行的容器失去 GPU 句柄。"
    echo "  建议修复:"
    echo "    docker restart $CONTAINER"
    echo "  修复后可继续执行，并配合 REEX_RESUME=1 从断点续跑。"
    exit 1
  fi
fi

DOCKER_ENV=()
for n in \
  REEX_GATE_PROFILE VERIFY_ROOT VERIFY_QUICK SKIP_CUDA VERIFY_JOBS VERIFY_GEMM_Q8 VERIFY_SKIP_RUNTIME \
  SKIP_TIER0 SKIP_TIER1 MODEL_F16 MODEL_Q4_K VERIFY_MODEL MODEL WIKITEXT_FILE VERIFY_WIKITEXT DATA \
  REEX_RESUME REEX_GATE_ALL_STATE_DIR VERIFY_CHUNKS VERIFY_CTX VERIFY_BATCH VERIFY_TIMEOUT_SEC \
  VERIFY_VARIANTS VERIFY_VARIANT_ORDER PROJECT_VARIANTS PROJECT_SMOKE_VARIANTS \
  VERIFY_THREADS VERIFY_NGL_CUDA VERIFY_NGL_CPU VERIFY_PPL_CTX VERIFY_PPL_BATCH VERIFY_PPL_CHUNKS \
  VERIFY_NGL VERIFY_Q8_E2E_CHUNKS VERIFY_Q8_E2E_KV_EXTRA VERIFY_Q8_E2E_SKIP_K8V4; do
  if [[ -n "${!n:-}" ]]; then
    DOCKER_ENV+=( -e "$n=${!n}" )
  fi
done

echo "=== run_reex_all_function_tests_docker.sh ==="
echo "  容器: $CONTAINER"
echo "  工作目录: $WORKDIR"
echo "  在容器内执行: ./scripts/run_reex_all_function_tests.sh $*"
echo ""

CMD="./scripts/run_reex_all_function_tests.sh"
for arg in "$@"; do
  CMD+=" $(printf '%q' "$arg")"
done

docker exec \
  "${DOCKER_ENV[@]}" \
  "$CONTAINER" \
  bash -lc "set -euo pipefail; cd '$WORKDIR' && exec $CMD"
