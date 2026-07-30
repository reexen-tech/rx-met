#!/usr/bin/env bash
# 首次初始化：校验、导入镜像、创建 workspace、生成 .env
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "${PKG_DIR}/.." && pwd)/workspace"
ENV_EXAMPLE="${PKG_DIR}/env.example"
ENV_FILE="${WORKSPACE}/.env"

log() { printf '[setup] %s\n' "$*"; }
die() { printf '[setup] ERROR: %s\n' "$*" >&2; exit 1; }

cd "${PKG_DIR}"

[[ -f SHA256SUMS ]] || die "SHA256SUMS not found in ${PKG_DIR}"
[[ -f "${ENV_EXAMPLE}" ]] || die "env.example not found: ${ENV_EXAMPLE}"

log "verify package checksums"
sha256sum -c SHA256SUMS

shopt -s nullglob
image_tars=("${PKG_DIR}"/rx-met-*-image.tar)
shopt -u nullglob
[[ ${#image_tars[@]} -eq 1 ]] || die "expected exactly one rx-met-*-image.tar in ${PKG_DIR}"
IMAGE_TAR="${image_tars[0]}"

IMAGE="$(grep -E '^IMAGE=' "${ENV_EXAMPLE}" | head -1 | cut -d= -f2-)"
[[ -n "${IMAGE}" ]] || die "IMAGE not set in env.example"
if [[ "${IMAGE}" == *"@VERSION@"* ]]; then
  die "IMAGE still contains @VERSION@; use a packaged release or substitute the version"
fi

ENABLE_GPUS="$(grep -E '^ENABLE_GPUS=' "${ENV_EXAMPLE}" | head -1 | cut -d= -f2- || true)"
ENABLE_GPUS="${ENABLE_GPUS:-1}"

log "docker load -> ${IMAGE_TAR}"
docker load -i "${IMAGE_TAR}"

smoke_args=(--rm)
if [[ "${ENABLE_GPUS}" == "1" ]]; then
  smoke_args+=(--gpus all)
fi
log "smoke: ${IMAGE} rx-met --help"
docker run "${smoke_args[@]}" "${IMAGE}" rx-met --help

log "prepare workspace -> ${WORKSPACE}"
mkdir -p "${WORKSPACE}/runs"
if [[ ! -d "${WORKSPACE}/examples" ]]; then
  cp -a "${PKG_DIR}/examples" "${WORKSPACE}/examples"
  log "copied examples -> ${WORKSPACE}/examples"
else
  log "examples already exist, skip copy: ${WORKSPACE}/examples"
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  cp "${ENV_EXAMPLE}" "${ENV_FILE}"
  if grep -qE '^WORKSPACE=' "${ENV_FILE}"; then
    sed -i "s|^WORKSPACE=.*|WORKSPACE=${WORKSPACE}|" "${ENV_FILE}"
  else
    printf 'WORKSPACE=%s\n' "${WORKSPACE}" >> "${ENV_FILE}"
  fi
  log "wrote ${ENV_FILE}"
else
  log ".env already exists, skip: ${ENV_FILE}"
fi

cat <<EOF

初始化完成。

下一步：
  1. 编辑 ${ENV_FILE}
       - MODELS_DIR    宿主机模型目录
       - DATASETS_DIR  宿主机数据集目录（可选）
  2. 按需编辑 ${WORKSPACE}/examples/config/llm_quant.json
  3. 启动容器：
       ${PKG_DIR}/scripts/rx-met-shell.sh

EOF
