#!/usr/bin/env bash
# Run inside ada200_docker:rx-met-dev. Copies the mounted source and builds wheels.
set -euo pipefail

SRC="${RX_MET_SRC:-/opt/rx-met-src}"
WORKDIR="${RX_MET_BUILD_COPY:-/tmp/rx-met-build}"
OUT="${RX_MET_WHEEL_OUT:-/artifacts/wheels}"

log() { printf '[build_wheels] %s\n' "$*"; }

[[ -d "${SRC}" ]] || { log "ERROR: source not mounted: ${SRC}"; exit 1; }
[[ -n "${RX_MET_VERSION:-}" ]] || { log "ERROR: RX_MET_VERSION is required"; exit 1; }

rm -rf "${WORKDIR}"
mkdir -p "${WORKDIR}" "${OUT}"
tar -C "${SRC}" \
    --exclude=.release \
    --exclude=.git \
    --exclude='*.egg-info' \
    --exclude=__pycache__ \
    -cf - . | tar -C "${WORKDIR}" -xf -

cd "${WORKDIR}"
python3 scripts/prepare_packaging.py --root "${WORKDIR}" --version "${RX_MET_VERSION}"

export RX_MET_ENABLE_CUDA=1
export RX_MET_PYTHON="${RX_MET_PYTHON:-python3}"
export RX_MET_WHEEL_OUT="${OUT}"
export RX_MET_AIMET_BUILD_DIR="${RX_MET_AIMET_BUILD_DIR:-/tmp/rx-met-aimet-build}"
export RX_MET_QUANT_GRU_BUILD_DIR="${RX_MET_QUANT_GRU_BUILD_DIR:-/tmp/rx-met-quant-gru-build}"

./scripts/build_wheel.sh
./scripts/build_quant_gru_wheel.sh
./scripts/download_extra_wheels.sh "${OUT}"

python3 - <<PY
from pathlib import Path
import zipfile

out = Path("${OUT}")
wheels = list(out.glob("rx_met-*.whl"))
assert len(wheels) == 1, wheels
names = zipfile.ZipFile(wheels[0]).namelist()
assert not any(name.startswith("rx_met_llm") for name in names), "rx_met_llm leaked into wheel"
print("wheel ok", wheels[0].name)
assert list(out.glob("quant_gru-*.whl")), "missing quant_gru wheel"
assert list(out.glob("torchaudio-*.whl")), "missing torchaudio wheel"
assert list(out.glob("librosa-*.whl")), "missing librosa wheel"
assert list(out.glob("onnxsim-*.whl")), "missing onnxsim wheel"
assert list(out.glob("tqdm-*.whl")), "missing tqdm wheel"
print("artifact wheels", sorted(p.name for p in out.glob("*.whl")))
PY
