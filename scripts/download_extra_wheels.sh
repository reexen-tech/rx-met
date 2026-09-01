#!/usr/bin/env bash
# Download offline extra wheels for the ada200_docker runtime (no torch / ORT).
set -euo pipefail

OUT_DIR="${1:?usage: download_extra_wheels.sh OUT_DIR}"
PYTHON="${RX_MET_PYTHON:-python3}"

mkdir -p "${OUT_DIR}"

log() { printf '[rx-met/download_extra_wheels] %s\n' "$*"; }

log "torchaudio 2.8.0 (cu128, no deps — torch is already in ada200_docker)"
"${PYTHON}" -m pip download \
    --dest "${OUT_DIR}" \
    --no-deps \
    --only-binary=:all: \
    --index-url https://download.pytorch.org/whl/cu128 \
    torchaudio==2.8.0

log "librosa + soundfile + onnxsim and their missing transitive wheels"
"${PYTHON}" -m pip download \
    --dest "${OUT_DIR}" \
    --only-binary=:all: \
    librosa==0.11.0 \
    soundfile==0.13.1 \
    onnxsim==0.7.0

# Keep the tarball from carrying a second copy of packages already in ada200_docker.
shopt -s nullglob
removed=0
for wheel in "${OUT_DIR}"/*.whl; do
    base="$(basename "${wheel}")"
    case "${base}" in
        torch-*|torchvision-*|nvidia-*|numpy-*|scipy-*|scikit_learn-*|scikit-learn-*|joblib-*|threadpoolctl-*|onnx-*|protobuf-*|ml_dtypes-*)
            rm -f "${wheel}"
            log "drop already-present runtime wheel: ${base}"
            removed=$((removed + 1))
            ;;
    esac
done
shopt -u nullglob
log "removed ${removed} wheels already provided by ada200_docker"
log "done: ${OUT_DIR}"
