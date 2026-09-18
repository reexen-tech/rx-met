#!/usr/bin/env bash
# 准备与 ONNX Runtime wheel API 完全一致的 AIMET custom-op 构建头文件。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERSION="${1:?用法: prepare_onnxruntime_headers.sh VERSION OUT_DIR SHA256}"
OUT_DIR="${2:?用法: prepare_onnxruntime_headers.sh VERSION OUT_DIR SHA256}"
SHA256="${3:?用法: prepare_onnxruntime_headers.sh VERSION OUT_DIR SHA256}"
VENDORED="${ROOT}/native/third_party/onnxruntime"

log() { printf '[prepare_onnxruntime_headers] %s\n' "$*"; }
die() { printf '[prepare_onnxruntime_headers] ERROR: %s\n' "$*" >&2; exit 1; }

[[ "${SHA256}" == "vendored" || "${SHA256}" =~ ^[0-9a-f]{64}$ ]] \
    || die "无效的源码 SHA-256: ${SHA256}"

rm -rf -- "${OUT_DIR}"
mkdir -p -- "${OUT_DIR}"

if [[ "${SHA256}" == "vendored" ]]; then
    (
        cd "${VENDORED}"
        sha256sum --check --strict --quiet SHA256SUMS
    )
    cp -a "${VENDORED}/." "${OUT_DIR}/"
else
    tmp="$(mktemp -d -t rx-met-onnxruntime-headers.XXXXXX)"
    cleanup() { rm -rf -- "${tmp}"; }
    trap cleanup EXIT
    archive="${RX_MET_ONNXRUNTIME_ARCHIVE:-${tmp}/onnxruntime.tar.gz}"
    source_root="${tmp}/source/onnxruntime-${VERSION}"
    url="https://github.com/microsoft/onnxruntime/archive/refs/tags/v${VERSION}.tar.gz"

    if [[ -n "${RX_MET_ONNXRUNTIME_ARCHIVE:-}" ]]; then
        [[ -f "${archive}" ]] || die "源码归档不存在: ${archive}"
        log "使用本地源码归档: ${archive}"
    else
        log "下载官方源码头文件: ${url}"
        curl -fL --retry 5 --connect-timeout 30 --max-time 1800 \
            --output "${archive}" "${url}"
    fi
    printf '%s  %s\n' "${SHA256}" "${archive}" | sha256sum --check --strict
    mkdir -p "${tmp}/source"
    tar -xzf "${archive}" -C "${tmp}/source" \
        "onnxruntime-${VERSION}/LICENSE" \
        "onnxruntime-${VERSION}/ThirdPartyNotices.txt" \
        "onnxruntime-${VERSION}/include/onnxruntime/core/session" \
        "onnxruntime-${VERSION}/include/onnxruntime/core/providers"

    mkdir -p "${OUT_DIR}/include/core"
    cp -a "${source_root}/include/onnxruntime/core/session/." "${OUT_DIR}/include/"
    cp -a "${source_root}/include/onnxruntime/core/providers" "${OUT_DIR}/include/core/"
    cp "${source_root}/LICENSE" "${source_root}/ThirdPartyNotices.txt" "${OUT_DIR}/"
    printf '%s\n' "${VERSION}" > "${OUT_DIR}/VERSION_NUMBER"
fi

actual="$(sed -n 's/^#define ORT_API_VERSION[[:space:]]\+//p' "${OUT_DIR}/include/onnxruntime_c_api.h")"
[[ "${actual}" =~ ^[0-9]+$ ]] || die "无法读取 ORT_API_VERSION"
log "完成: version=${VERSION}, api=${actual}, path=${OUT_DIR}"
