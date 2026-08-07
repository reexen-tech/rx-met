#!/usr/bin/env bash
# Build the rx-met wheel only (deps installed in Dockerfile).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${RX_MET_WHEEL_OUT:-${ROOT}/.release/wheels}"
PYTHON="${RX_MET_PYTHON:-python3}"

log() { printf '[build_wheel] %s\n' "$*"; }

mkdir -p "${OUT_DIR}"
rm -f "${OUT_DIR}"/rx_met-*.whl "${OUT_DIR}"/rx-met-*.whl \
    "${OUT_DIR}"/aimet_rx-*.whl "${OUT_DIR}"/aimet-rx-*.whl

log "build AIMET native runtime from repository source"
"${ROOT}/scripts/build_aimet_native.sh"

log "pip wheel (no deps) -> ${OUT_DIR}"
"${PYTHON}" -m pip wheel "${ROOT}" -w "${OUT_DIR}" --no-deps --no-build-isolation

wheel=("${OUT_DIR}"/rx_met-*.whl)
if ((${#wheel[@]} != 1)) || [[ ! -f "${wheel[0]}" ]]; then
    log "ERROR: expected exactly one rx_met wheel"
    exit 1
fi
if [[ "$(basename "${wheel[0]}")" != *-cp310-cp310-linux_x86_64.whl ]]; then
    log "ERROR: expected a CPython 3.10 Linux x86_64 wheel, got ${wheel[0]}"
    exit 1
fi
log "done: ${wheel[0]}"
