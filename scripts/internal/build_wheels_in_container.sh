#!/usr/bin/env bash
# 在 NVIDIA CUDA devel builder 中构建 rx-met 和 QuantGRU wheel。
set -euo pipefail

SRC="${RX_MET_SRC:-/opt/rx-met-src}"
WORKDIR="${RX_MET_BUILD_COPY:-/tmp/rx-met-build}"
OUT="${RX_MET_WHEEL_OUT:-/artifacts/wheels}"

log() { printf '[build_wheels] %s\n' "$*"; }

[[ -d "${SRC}" ]] || { log "ERROR: source not mounted: ${SRC}"; exit 1; }
[[ -n "${RX_MET_VERSION:-}" ]] || { log "ERROR: RX_MET_VERSION is required"; exit 1; }
[[ -n "${CUDA_VARIANT:-}" ]] || { log "ERROR: CUDA_VARIANT is required"; exit 1; }
for required in \
    pyproject.toml \
    native/aimet/CMakeLists.txt \
    quant-gru/CMakeLists.txt \
    quant-gru/pytorch/_version.py
do
    [[ -f "${SRC}/${required}" ]] \
        || { log "ERROR: source file missing: ${SRC}/${required}"; exit 1; }
done

rm -rf "${WORKDIR}"
mkdir -p "${WORKDIR}" "${OUT}"
tar -C "${SRC}" \
    --exclude=.release \
    --exclude=.git \
    --exclude='*.egg-info' \
    --exclude=__pycache__ \
    -cf - . | tar -C "${WORKDIR}" -xf -

cd "${WORKDIR}"
python3 scripts/internal/prepare_packaging.py \
    --root "${WORKDIR}" \
    --version "${RX_MET_VERSION}" \
    --cuda-variant "${CUDA_VARIANT}"

export RX_MET_ENABLE_CUDA=1
export RX_MET_PYTHON="${RX_MET_PYTHON:-python3}"
export RX_MET_WHEEL_OUT="${OUT}"
export RX_MET_AIMET_BUILD_DIR="${RX_MET_AIMET_BUILD_DIR:-/tmp/rx-met-aimet-build}"
export RX_MET_QUANT_GRU_BUILD_DIR="${RX_MET_QUANT_GRU_BUILD_DIR:-/tmp/rx-met-quant-gru-build}"

rm -f "${OUT}"/rx_met-*.whl "${OUT}"/rx-met-*.whl \
    "${OUT}"/aimet_rx-*.whl "${OUT}"/aimet-rx-*.whl
log "build AIMET native runtime from repository source"
./scripts/build_aimet_native.sh
log "build rx-met wheel -> ${OUT}"
"${RX_MET_PYTHON}" -m pip wheel "${WORKDIR}" \
    -w "${OUT}" --no-deps --no-build-isolation
./scripts/internal/build_quant_gru_wheel.sh

python3 - <<PY
from pathlib import Path
import sys
import zipfile

out = Path("${OUT}")
wheels = list(out.glob("rx_met-*.whl"))
assert len(wheels) == 1, wheels
python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
assert wheels[0].name.endswith(
    f"-{python_tag}-{python_tag}-linux_x86_64.whl"
), wheels[0]
names = zipfile.ZipFile(wheels[0]).namelist()
assert not any(name.startswith("rx_met_llm") for name in names), "rx_met_llm leaked into wheel"
print("wheel ok", wheels[0].name)
assert list(out.glob("quant_gru-*.whl")), "missing quant_gru wheel"
print("artifact wheels", sorted(p.name for p in out.glob("*.whl")))
PY
