#!/usr/bin/env bash
# 下载当前 CUDA 变体的完整离线 wheelhouse。仅在 Docker builder 中运行。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${1:?用法: download_runtime_wheels.sh OUT_DIR CUDA_VARIANT}"
CUDA_VARIANT="${2:?用法: download_runtime_wheels.sh OUT_DIR CUDA_VARIANT}"
PYTHON="${RX_MET_PYTHON:-python3}"
COMMON_LOCK="${ROOT}/docker/requirements/common.lock"
BUILD_LOCK="${ROOT}/docker/requirements/build.lock"
VARIANT_LOCK="${ROOT}/docker/requirements/${CUDA_VARIANT}.txt"
VERIFY_SCRIPT="${ROOT}/scripts/internal/verify_dependency_wheelhouse.py"

log() { printf '[download_runtime_wheels] %s\n' "$*"; }
die() { printf '[download_runtime_wheels] ERROR: %s\n' "$*" >&2; exit 1; }

case "${CUDA_VARIANT}" in
    cu118|cu126|cu130) ;;
    *) die "不支持的 CUDA 变体: ${CUDA_VARIANT}" ;;
esac
[[ -f "${COMMON_LOCK}" ]] || die "缺少公共锁文件: ${COMMON_LOCK}"
[[ -f "${BUILD_LOCK}" ]] || die "缺少构建工具锁文件: ${BUILD_LOCK}"
[[ -f "${VARIANT_LOCK}" ]] || die "缺少变体锁文件: ${VARIANT_LOCK}"
[[ -f "${VERIFY_SCRIPT}" ]] || die "缺少 wheelhouse 校验器: ${VERIFY_SCRIPT}"

mkdir -p "${OUT_DIR}"

verify_offline_wheelhouse() {
    "${PYTHON}" "${VERIFY_SCRIPT}" \
        "${OUT_DIR}" "${BUILD_LOCK}" "${COMMON_LOCK}" "${VARIANT_LOCK}"
}

if compgen -G "${OUT_DIR}/*.whl" >/dev/null \
    && verify_offline_wheelhouse; then
    log "现有 wheel 已完整满足锁文件，跳过网络下载"
    (
        cd "${OUT_DIR}"
        sha256sum ./*.whl | sort -k2 > SHA256SUMS
    )
    exit 0
fi

log "下载 Python 构建工具"
"${PYTHON}" -m pip download \
    --dest "${OUT_DIR}" \
    --only-binary=:all: \
    --no-deps \
    -r "${BUILD_LOCK}"

log "下载公共依赖"
"${PYTHON}" -m pip download \
    --dest "${OUT_DIR}" \
    --only-binary=:all: \
    --no-deps \
    -r "${COMMON_LOCK}"

log "下载 ${CUDA_VARIANT} 的 PyTorch/CUDA 依赖"
"${PYTHON}" -m pip download \
    --dest "${OUT_DIR}" \
    --only-binary=:all: \
    --no-deps \
    -r "${VARIANT_LOCK}"

verify_offline_wheelhouse \
    || die "下载后的 wheelhouse 无法离线满足锁文件: ${CUDA_VARIANT}"

(
    cd "${OUT_DIR}"
    sha256sum ./*.whl | sort -k2 > SHA256SUMS
)
log "完成: ${OUT_DIR}"
