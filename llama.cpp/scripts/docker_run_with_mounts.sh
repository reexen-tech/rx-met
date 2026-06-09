#!/bin/bash
# 宿主机执行：用当前目录挂载为 /workspace/llama.cpp，并挂载 datasets 与 models 目录。
# 挂载目录变了时，只需改下面两个变量（或 export 环境变量）后重新运行本脚本。
#
# 用法:
#   ./scripts/docker_run_with_mounts.sh [镜像名]
# 示例:
#   export LLAMA_DATASETS_HOST=/data/share/datasets
#   export LLAMA_MODELS_HOST=/data/share/models
#   ./scripts/docker_run_with_mounts.sh
#   # 或直接写路径：
#   LLAMA_DATASETS_HOST=/data/datasets LLAMA_MODELS_HOST=/data/models ./scripts/docker_run_with_mounts.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 宿主机上 datasets 与 models 的路径（挂载目录变了就改这里或用环境变量）
DATASETS_HOST="${LLAMA_DATASETS_HOST:-/mnt/data8t/share/datasets}"
MODELS_HOST="${LLAMA_MODELS_HOST:-/mnt/data8t/share/models}"

# 容器内挂载点（与 remount_datasets_models.sh 中符号链接目标一致）
DATASETS_MOUNT="/mnt/data8t/share/datasets"
MODELS_MOUNT="/mnt/data8t/share/models"

IMAGE="${1:-ghcr.io/ggml-org/llama.cpp:full}"

if [ ! -d "$DATASETS_HOST" ]; then
  echo "警告: 宿主机目录不存在: $DATASETS_HOST" >&2
  echo "请设置 LLAMA_DATASETS_HOST 或修改本脚本中的默认值" >&2
fi
if [ ! -d "$MODELS_HOST" ]; then
  echo "警告: 宿主机目录不存在: $MODELS_HOST" >&2
  echo "请设置 LLAMA_MODELS_HOST 或修改本脚本中的默认值" >&2
fi

echo "挂载: $DATASETS_HOST -> $DATASETS_MOUNT"
echo "挂载: $MODELS_HOST -> $MODELS_MOUNT"
echo "挂载: $LLAMA_ROOT -> /workspace/llama.cpp"
echo "镜像: $IMAGE"
echo ""

docker run -it --rm \
  -v "$DATASETS_HOST:$DATASETS_MOUNT:ro" \
  -v "$MODELS_HOST:$MODELS_MOUNT:ro" \
  -v "$LLAMA_ROOT:/workspace/llama.cpp" \
  -w /workspace/llama.cpp \
  "$IMAGE" \
  bash
