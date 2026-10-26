#!/usr/bin/env bash
# Build the quant_gru wheel for the selected CPU or CUDA runtime.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="${ROOT}/operators/quant-gru"
PYTHON="${RX_MET_PYTHON:-python3}"
BUILD_DIR="${RX_MET_QUANT_GRU_BUILD_DIR:-${SOURCE}/build_release}"
OUT_DIR="${RX_MET_WHEEL_OUT:-${ROOT}/.release/wheels}"

log() { printf '[build_quant_gru_wheel] %s\n' "$*"; }

ENABLE_CUDA="${RX_MET_ENABLE_CUDA:-1}"
case "${ENABLE_CUDA}" in
    0) cmake_cuda=OFF ;;
    1) cmake_cuda=ON ;;
    *) log "ERROR: RX_MET_ENABLE_CUDA must be 0 or 1"; exit 1 ;;
esac

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    log "ERROR: ${PYTHON} is required"
    exit 1
fi
case "$("${PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" in
    3.10|3.11|3.12) ;;
    *) log "ERROR: quant_gru wheel requires Python 3.10-3.12"; exit 1 ;;
esac
if [[ "${ENABLE_CUDA}" == "1" ]]; then
    command -v nvcc >/dev/null 2>&1 \
        || { log "ERROR: CUDA Toolkit/nvcc is required to build quant_gru"; exit 1; }
    "${PYTHON}" -c \
        'import torch; assert torch.version.cuda is not None, "CUDA-enabled PyTorch is required"'
fi

log "build native library -> ${BUILD_DIR}"
cmake_args=(-DCMAKE_BUILD_TYPE=Release "-DRX_MET_ENABLE_CUDA=${cmake_cuda}")
if [[ "${ENABLE_CUDA}" == "1" && -n "${RX_MET_CUDA_ARCHITECTURES:-}" ]]; then
    cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=${RX_MET_CUDA_ARCHITECTURES}")
fi
cmake -S "${SOURCE}" -B "${BUILD_DIR}" "${cmake_args[@]}"
cmake --build "${BUILD_DIR}" -j "${JOBS:-$(nproc)}"

rm -f "${OUT_DIR}"/quant_gru-*.whl
mkdir -p "${OUT_DIR}"
log "build wheel -> ${OUT_DIR}"
export RX_MET_ENABLE_CUDA="${ENABLE_CUDA}"
export RX_MET_QUANT_GRU_BUILD_DIR="${BUILD_DIR}"
if [[ -n "${CUDA_VARIANT:-}" ]]; then
    export RX_MET_QUANT_GRU_VERSION="${RX_MET_VERSION:?RX_MET_VERSION is required}+${CUDA_VARIANT}"
fi
"${PYTHON}" -m pip wheel "${SOURCE}/pytorch" \
    -w "${OUT_DIR}" --no-deps --no-build-isolation

wheel=("${OUT_DIR}"/quant_gru-*.whl)
if ((${#wheel[@]} != 1)) || [[ ! -f "${wheel[0]}" ]]; then
    log "ERROR: expected exactly one quant_gru wheel"
    exit 1
fi
log "done: ${wheel[0]}"
