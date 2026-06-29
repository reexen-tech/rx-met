#!/usr/bin/env bash
# One-click build of the reex-gemm-datagen tool.
#   - configures BUILD_DIR with the required reex/CUDA flags if it does not exist
#   - builds the reex-gemm-datagen target
#
# Env overrides:
#   BUILD_DIR                       cmake build dir         (default build_cuda_q64)
#   REEX_GEMM_DATAGEN_CUDA_ARCH     CUDA arch               (default 120, RTX5090/Blackwell)
#   JOBS                            parallel build jobs     (default nproc)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

CUDA_ARCH="${REEX_GEMM_DATAGEN_CUDA_ARCH:-120}"
JOBS="${JOBS:-$(nproc)}"

cd "$LLAMA_ROOT"

if [[ ! -d "$BUILD_DIR" ]]; then
    echo "==> configuring $BUILD_DIR (CUDA arch $CUDA_ARCH)"
    cmake -S . -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON -DGGML_USE_REEX_Q64=ON -DGGML_REEX_FP16_PIPELINE=ON -DGGML_USE_REEX=ON \
        -DREEX_GEMM_DATAGEN=ON -DREEX_GEMM_DATAGEN_CUDA_ARCH="$CUDA_ARCH"
fi

echo "==> building reex-gemm-datagen (-j $JOBS)"
cmake --build "$BUILD_DIR" --target reex-gemm-datagen -j "$JOBS"

echo "==> done: $BIN"
