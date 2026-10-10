#!/usr/bin/env bash
# Build the CUDA-backed quant_lstm wheel for the runtime Python ABI.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="${ROOT}/operators/quant-lstm"
PYTHON="${RX_MET_PYTHON:-python3}"
BUILD_DIR="${RX_MET_QUANT_LSTM_BUILD_DIR:-${SOURCE}/build_release}"
OUT_DIR="${RX_MET_WHEEL_OUT:-${ROOT}/.release/wheels}"

log() { printf '[build_quant_lstm_wheel] %s\n' "$*"; }

if [[ "${RX_MET_ENABLE_CUDA:-1}" != "1" ]]; then
    log "SKIP: quant_lstm requires CUDA (RX_MET_ENABLE_CUDA=${RX_MET_ENABLE_CUDA:-0}). Use the GPU release for QuantLSTM."
    exit 0
fi

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    log "ERROR: ${PYTHON} is required"
    exit 1
fi
case "$("${PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" in
    3.10|3.11|3.12) ;;
    *) log "ERROR: quant_lstm wheel requires Python 3.10-3.12"; exit 1 ;;
esac
if ! command -v nvcc >/dev/null 2>&1; then
    log "ERROR: CUDA Toolkit/nvcc is required to build quant_lstm"
    exit 1
fi
"${PYTHON}" -c \
    'import torch; assert torch.version.cuda is not None, "CUDA-enabled PyTorch is required"'

log "build native library -> ${BUILD_DIR}"
cmake_args=(-DCMAKE_BUILD_TYPE=Release -DQUANT_LSTM_ENABLE_CUDA=ON
    -DQUANT_LSTM_BUILD_TESTS=OFF -DQUANT_LSTM_BUILD_EXAMPLES=OFF)
if [[ -n "${RX_MET_CUDA_ARCHITECTURES:-}" ]]; then
    cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=${RX_MET_CUDA_ARCHITECTURES}")
fi
cmake -S "${SOURCE}" -B "${BUILD_DIR}" "${cmake_args[@]}"
cmake --build "${BUILD_DIR}" -j "${JOBS:-$(nproc)}"

rm -f "${OUT_DIR}"/quant_lstm-*.whl
mkdir -p "${OUT_DIR}"
log "build wheel -> ${OUT_DIR}"
export RX_MET_QUANT_LSTM_BUILD_DIR="${BUILD_DIR}"
if [[ -n "${CUDA_VARIANT:-}" ]]; then
    export RX_MET_QUANT_LSTM_VERSION="0.1.0+${CUDA_VARIANT}"
fi
"${PYTHON}" -m pip wheel "${SOURCE}/pytorch" \
    -w "${OUT_DIR}" --no-deps --no-build-isolation

wheel=("${OUT_DIR}"/quant_lstm-*.whl)
if ((${#wheel[@]} != 1)) || [[ ! -f "${wheel[0]}" ]]; then
    log "ERROR: expected exactly one quant_lstm wheel"
    exit 1
fi
log "done: ${wheel[0]}"
