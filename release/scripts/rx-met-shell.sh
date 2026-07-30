#!/usr/bin/env bash
# 按 workspace/.env 启动已配置好的 rx-met 容器。
# 用法：
#   ./scripts/rx-met-shell.sh              # 进入交互 shell
#   ./scripts/rx-met-shell.sh -- <cmd...>  # 在容器内执行命令
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_WORKSPACE="$(cd "${PKG_DIR}/.." && pwd)/workspace"

die() { printf '[rx-met-shell] ERROR: %s\n' "$*" >&2; exit 1; }

ENV_FILE="${DEFAULT_WORKSPACE}/.env"
[[ -f "${ENV_FILE}" ]] || die "missing ${ENV_FILE}; run scripts/setup.sh first"

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

WORKSPACE="${WORKSPACE:-${DEFAULT_WORKSPACE}}"
IMAGE="${IMAGE:-}"
[[ -n "${IMAGE}" ]] || die "IMAGE not set in ${ENV_FILE}"
[[ -d "${WORKSPACE}" ]] || die "WORKSPACE is not a directory: ${WORKSPACE}"

docker_args=(
  --rm
  --gpus all
  --user "$(id -u):$(id -g)"
  -e HOME=/tmp
  -e USER="$(id -un)"
  -e RX_MET_SPEECH_COMMANDS_ROOT=/datasets/speech_commands_v0.02
  -v "${WORKSPACE}:/workspace"
  -w /workspace
)

if [[ -n "${MODELS_DIR:-}" ]]; then
  [[ -d "${MODELS_DIR}" ]] || die "MODELS_DIR is not a directory: ${MODELS_DIR}"
  MODELS_DIR="$(cd "${MODELS_DIR}" && pwd)"
  docker_args+=(-v "${MODELS_DIR}:/models:ro")
fi

if [[ -n "${DATASETS_DIR:-}" ]]; then
  [[ -d "${DATASETS_DIR}" ]] || die "DATASETS_DIR is not a directory: ${DATASETS_DIR}"
  DATASETS_DIR="$(cd "${DATASETS_DIR}" && pwd)"
  docker_args+=(-v "${DATASETS_DIR}:/datasets:ro")
fi

if [[ -t 0 && -t 1 ]]; then
  docker_args+=(-it)
fi

if [[ $# -eq 0 ]]; then
  exec docker run "${docker_args[@]}" "${IMAGE}" bash
fi

if [[ "$1" == "--" ]]; then
  shift
fi
[[ $# -gt 0 ]] || die "missing command after --"
exec docker run "${docker_args[@]}" "${IMAGE}" "$@"
