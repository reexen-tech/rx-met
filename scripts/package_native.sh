#!/usr/bin/env bash
# Build llama.cpp (REEX Q64; CUDA or CPU) and pack bin/ + lib/ into native/*.tar.gz.
#
# RX_MET_ENABLE_CUDA=1|0   是否开启 GGML CUDA（默认 1）
# RX_MET_CUDA_VERSION      CUDA 版本号（如 12.8）；ENABLE_CUDA=1 时必填，用于产物命名
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA="${ROOT}/llama.cpp"

PRODUCT_VERSION="$(tr -d '[:space:]' < "${ROOT}/VERSION")"
VERSION="${RX_MET_VERSION:-${PRODUCT_VERSION}}"
ENABLE_CUDA="${RX_MET_ENABLE_CUDA:-1}"
if [[ "${ENABLE_CUDA}" == "1" ]]; then
    if [[ -z "${RX_MET_CUDA_VERSION:-}" ]]; then
        printf '[package_native] ERROR: RX_MET_CUDA_VERSION is required when RX_MET_ENABLE_CUDA=1\n' >&2
        exit 1
    fi
    PLATFORM_TAG="cuda${RX_MET_CUDA_VERSION}"
else
    PLATFORM_TAG="cpu"
fi
BUILD_DIR="${RX_MET_LLAMA_BUILD_DIR:-${LLAMA}/build_release}"
OUT_DIR="${RX_MET_NATIVE_OUT:-${ROOT}/.release/native}"
TARBALL="${OUT_DIR}/rx-met-native-${VERSION}-${PLATFORM_TAG}-linux_x86_64.tar.gz"

if [[ "$(uname -m)" != "x86_64" ]]; then
    printf '[package_native] ERROR: release target is x86_64, host is %s\n' "$(uname -m)"
    exit 1
fi
if [[ -n "${RX_MET_NATIVE_STAGE:-}" ]]; then
    STAGE="${RX_MET_NATIVE_STAGE}"
    STAGE_OWNED=0
else
    STAGE="$(mktemp -d -t rx-met-native-stage.XXXXXX)"
    STAGE_OWNED=1
fi
cleanup() {
    if [[ "${STAGE_OWNED}" == "1" ]]; then
        rm -rf "${STAGE}"
    fi
}
trap cleanup EXIT

REQUIRED_BIN=(
    llama-quantize
    llama-imatrix
    llama-perplexity
    reex-hw-convert
    convert_hf_to_gguf.py
)

log() { printf '[package_native] %s\n' "$*"; }

if [[ "${VERSION}" != "${PRODUCT_VERSION}" ]]; then
    log "ERROR: RX_MET_VERSION=${VERSION} differs from ${ROOT}/VERSION (${PRODUCT_VERSION})"
    exit 1
fi

if [[ "${SKIP_LLAMA_BUILD:-0}" != "1" ]]; then
    log "configure + build -> ${BUILD_DIR} (platform=${PLATFORM_TAG}, enable_cuda=${ENABLE_CUDA})"
    cmake_args=(
        -DCMAKE_BUILD_TYPE=Release
        -DGGML_USE_REEX_Q64=ON
        -DGGML_USE_REEX=ON
        -DLLAMA_BUILD_TOOLS=ON
        -DLLAMA_TOOLS_INSTALL=ON
        -DREEX_HW_CONVERT=ON
    )
    if [[ "${ENABLE_CUDA}" == "1" ]]; then
        cmake_args+=(-DGGML_CUDA=ON)
        if [[ -n "${RX_MET_CUDA_ARCHITECTURES:-}" ]]; then
            cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=${RX_MET_CUDA_ARCHITECTURES}")
        fi
    else
        cmake_args+=(-DGGML_CUDA=OFF)
    fi
    cmake -S "${LLAMA}" -B "${BUILD_DIR}" "${cmake_args[@]}"
    cmake --build "${BUILD_DIR}" -j "${JOBS:-$(nproc)}"
else
    log "SKIP_LLAMA_BUILD=1, using existing ${BUILD_DIR}"
fi

mkdir -p "${STAGE}"
rm -rf "${STAGE:?}/"*
log "install -> ${STAGE}"
if [[ "${RX_MET_STRIP_NATIVE:-1}" == "1" ]]; then
    cmake --install "${BUILD_DIR}" --prefix "${STAGE}" --strip
else
    cmake --install "${BUILD_DIR}" --prefix "${STAGE}"
fi

BIN="${STAGE}/bin"
LIB="${STAGE}/lib"
mkdir -p "${BIN}" "${LIB}"

# Some in-tree builds drop shared libs next to binaries; normalize to lib/.
shopt -s nullglob
for so in "${BIN}"/*.so*; do
    base="$(basename "${so}")"
    if [[ -L "${LIB}/${base}" || -e "${LIB}/${base}" ]]; then
        rm -f "${so}"
    else
        mv "${so}" "${LIB}/"
    fi
done
shopt -u nullglob

missing=()
for name in "${REQUIRED_BIN[@]}"; do
    if [[ ! -x "${BIN}/${name}" ]]; then
        missing+=("${name}")
    fi
done
if ((${#missing[@]} > 0)); then
    log "ERROR: missing required tools in ${BIN}:"
    printf '  - %s\n' "${missing[@]}"
    exit 1
fi

for name in llama-quantize llama-imatrix llama-perplexity reex-hw-convert; do
    if LD_LIBRARY_PATH="${LIB}:${LD_LIBRARY_PATH:-}" \
        ldd "${BIN}/${name}" | grep -q 'not found'; then
        log "ERROR: unresolved shared libraries for ${BIN}/${name}:"
        LD_LIBRARY_PATH="${LIB}:${LD_LIBRARY_PATH:-}" \
            ldd "${BIN}/${name}" | grep 'not found'
        exit 1
    fi
done

mkdir -p "${OUT_DIR}"
log "tar -> ${TARBALL}"
tar -C "${STAGE}" -czf "${TARBALL}" bin lib
log "done: ${TARBALL} ($(du -h "${TARBALL}" | cut -f1))"
