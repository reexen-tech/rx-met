#!/usr/bin/env bash
# === REEX_SMOOTHQUANT BEGIN: host-side wrapper running the experiment in a CUDA container ===
# 宿主机执行：在已有 CUDA 容器内跑 SmoothQuant + 指定量化类型的端到端实验。
# 依赖：容器内已挂载本仓库到 REEX_SQ_WORKDIR，并已 cmake build 出 llama-quantize / llama-perplexity。
#
# 用法（宿主机，仓库根目录）:
#   ./llm_quant/smoothquant/scripts/run_smoothquant_experiment_docker.sh \
#     --model_path /mnt/data8t/share/models/Qwen/Qwen2.5-7B-Instruct \
#     --output_dir runs/qwen2.5-7b-sq-q8_64 \
#     --calib_data /path/to/calib.jsonl \
#     --ppl_dataset /path/to/wiki.test.raw \
#     --quant-type Q4_K_64
#
#   REEX_SQ_CONTAINER=my-cuda REEX_SQ_WORKDIR=/workspace/llama.cpp \
#     ./llm_quant/smoothquant/scripts/run_smoothquant_experiment_docker.sh ...
set -euo pipefail

CONTAINER="${REEX_SQ_CONTAINER:-quant-gru-cuda128}"
WORKDIR="${REEX_SQ_WORKDIR:-/workspace/aimet_rx/llama.cpp}"

if ! docker info >/dev/null 2>&1; then
  echo "错误: 无法连接 Docker daemon。" >&2
  exit 1
fi
if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "错误: 容器「$CONTAINER」未在运行。" >&2
  echo "  先启动: docker start $CONTAINER" >&2
  echo "  或指定容器名: REEX_SQ_CONTAINER=<name> $0 ..." >&2
  echo "  当前运行中的候选容器:" >&2
  docker ps --format '    {{.Names}}' >&2
  exit 1
fi

# 实验脚本会把容器名写进 experiment_report.md 的环境信息里
ARGS=( "$@" "--container" "$CONTAINER" )
QUOTED=$(printf ' %q' "${ARGS[@]}")

echo "=== run_smoothquant_experiment_docker.sh ==="
echo "  容器: $CONTAINER"
echo "  工作目录: $WORKDIR"
echo "  命令: python llm_quant/smoothquant/scripts/run_smoothquant_experiment.py$QUOTED"
echo ""

docker exec "$CONTAINER" bash -lc \
  "set -euo pipefail; cd '$WORKDIR' && exec python llm_quant/smoothquant/scripts/run_smoothquant_experiment.py$QUOTED"
# === REEX_SMOOTHQUANT END ===
