#!/usr/bin/env bash
# Build the CUDA-backed quant_gru wheel for the runtime Python ABI.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${ROOT}/quant-gru-pytorch"
PYTHON="${RX_MET_PYTHON:-python3.10}"
BUILD_DIR="${RX_MET_QUANT_GRU_BUILD_DIR:-${SOURCE}/build_release}"
OUT_DIR="${RX_MET_WHEEL_OUT:-${ROOT}/.release/wheels}"

log() { printf '[build_quant_gru_wheel] %s\n' "$*"; }

if [[ "${RX_MET_ENABLE_CUDA:-1}" != "1" ]]; then
    log "SKIP: quant_gru requires CUDA (RX_MET_ENABLE_CUDA=${RX_MET_ENABLE_CUDA:-0}). Use the GPU release for QuantGRU / quick_start.py."
    exit 0
fi

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    log "ERROR: ${PYTHON} is required (runtime image uses Python 3.10)"
    exit 1
fi
if [[ "$("${PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.10" ]]; then
    log "ERROR: quant_gru wheel must be built for Python 3.10"
    exit 1
fi
if ! command -v nvcc >/dev/null 2>&1; then
    log "ERROR: CUDA Toolkit/nvcc is required to build quant_gru"
    exit 1
fi
"${PYTHON}" -c \
    'import torch; assert torch.version.cuda is not None, "CUDA-enabled PyTorch is required"'

log "build native library -> ${BUILD_DIR}"
cmake_args=(-DCMAKE_BUILD_TYPE=Release)
if [[ -n "${RX_MET_CUDA_ARCHITECTURES:-}" ]]; then
    cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=${RX_MET_CUDA_ARCHITECTURES}")
fi
cmake -S "${SOURCE}" -B "${BUILD_DIR}" "${cmake_args[@]}"
cmake --build "${BUILD_DIR}" -j "${JOBS:-$(nproc)}"

rm -f "${OUT_DIR}"/quant_gru-*.whl
mkdir -p "${OUT_DIR}"
log "build wheel -> ${OUT_DIR}"
"${PYTHON}" -m pip wheel "${SOURCE}/pytorch" \
    -w "${OUT_DIR}" --no-deps --no-build-isolation

wheel=("${OUT_DIR}"/quant_gru-*.whl)
if ((${#wheel[@]} != 1)) || [[ ! -f "${wheel[0]}" ]]; then
    log "ERROR: expected exactly one quant_gru wheel"
    exit 1
fi
log "done: ${wheel[0]}"
