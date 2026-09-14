#!/usr/bin/env bash
# 准备与 ONNX Runtime wheel API 完全一致的 AIMET custom-op 构建头文件。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERSION="${1:?用法: prepare_onnxruntime_headers.sh VERSION OUT_DIR}"
OUT_DIR="${2:?用法: prepare_onnxruntime_headers.sh VERSION OUT_DIR}"
VENDORED="${ROOT}/native/aimet/third_party/onnxruntime"

log() { printf '[prepare_onnxruntime_headers] %s\n' "$*"; }
die() { printf '[prepare_onnxruntime_headers] ERROR: %s\n' "$*" >&2; exit 1; }

case "${VERSION}" in
    1.20.1) SHA256="d4c005506a2bbf88a838b14f8d1578406b8be2fb64abb50beeff908fb272529e" ;;
    1.23.2) SHA256="vendored" ;;
    1.27.0) SHA256="b41d09905a3c2f3a25709d1dcce8ef3942a4c2799d1046f74be7b6bbebc45e6a" ;;
    *) die "不支持的 ONNX Runtime 版本: ${VERSION}" ;;
esac

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
