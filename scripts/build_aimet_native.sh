#!/usr/bin/env bash
# Build the rx-met AIMET native runtime from repository source.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${ROOT}/native/aimet"
PYTHON="${RX_MET_PYTHON:-python3}"
ENABLE_CUDA="${RX_MET_ENABLE_CUDA:-0}"
INSTALL_DIR="${RX_MET_AIMET_INSTALL_DIR:-${ROOT}/aimet_common}"

log() { printf '[build_aimet_native] %s\n' "$*"; }
fail() {
    log "ERROR: $*"
    exit 1
}

case "${ENABLE_CUDA}" in
    0) variant=cpu ;;
    1) variant=cuda ;;
    *) fail "RX_MET_ENABLE_CUDA must be 0 or 1, got ${ENABLE_CUDA}" ;;
esac

BUILD_DIR="${RX_MET_AIMET_BUILD_DIR:-${ROOT}/.release/build/aimet-${variant}}"

runtime="$("${PYTHON}" -c 'import platform, sys; print(sys.implementation.name, sys.version_info.major, sys.version_info.minor, platform.system(), platform.machine())')"
if [[ "${runtime}" != "cpython 3 10 Linux x86_64" ]]; then
    fail "requires CPython 3.10 on Linux x86_64; found ${runtime}"
fi

log "verify vendored ONNX Runtime 1.23.2 headers"
(
    cd "${SOURCE_DIR}/third_party/onnxruntime"
    sha256sum --check --strict --quiet SHA256SUMS
)

pybind11_dir="$("${PYTHON}" -m pybind11 --cmakedir)"
cmake_args=(
    -S "${SOURCE_DIR}"
    -B "${BUILD_DIR}"
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}"
    -DPython3_EXECUTABLE="$(command -v "${PYTHON}")"
    -Dpybind11_DIR="${pybind11_dir}"
    -DRX_MET_AIMET_ENABLE_CUDA="$([[ "${ENABLE_CUDA}" == "1" ]] && printf ON || printf OFF)"
)

if [[ -n "${RX_MET_ONNXRUNTIME_ROOT:-}" ]]; then
    cmake_args+=("-DRX_MET_ONNXRUNTIME_ROOT=${RX_MET_ONNXRUNTIME_ROOT}")
fi
if [[ "${ENABLE_CUDA}" == "1" && -n "${RX_MET_CUDA_ARCHITECTURES:-}" ]]; then
    cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=${RX_MET_CUDA_ARCHITECTURES}")
fi

log "configure ${variant} -> ${BUILD_DIR}"
cmake "${cmake_args[@]}"
log "compile from source"
cmake --build "${BUILD_DIR}" --parallel "${JOBS:-$(nproc)}"
log "install -> ${INSTALL_DIR}"
cmake --install "${BUILD_DIR}"

native_files=(
    "${INSTALL_DIR}/_libpymo.cpython-310-x86_64-linux-gnu.so"
    "${INSTALL_DIR}/libquant_info.cpython-310-x86_64-linux-gnu.so"
    "${INSTALL_DIR}/libaimet_onnxrt_ops.so"
)
for native_file in "${native_files[@]}"; do
    [[ -s "${native_file}" ]] || fail "missing build output: ${native_file}"
done

log "built AIMET native runtime (${variant})"
