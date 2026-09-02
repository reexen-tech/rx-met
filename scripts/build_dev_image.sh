#!/usr/bin/env bash
# Build ada200_docker:rx-met-dev from official ada200_docker, or from Dockerfile.ada200.
# Does not replace ada200_docker:latest.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADA200_IMAGE="${RX_MET_ADA200_IMAGE:-ada200_docker:latest}"
DEV_IMAGE="${RX_MET_DEV_IMAGE:-ada200_docker:rx-met-dev}"

log() { printf '[build_dev_image] %s\n' "$*"; }

rx_met_require_image_name() {
    local name="$1"
    if [[ ! "${name}" =~ ^[A-Za-z0-9._/-]+(:[A-Za-z0-9._-]+)?$ ]] \
        && [[ ! "${name}" =~ ^sha256:[a-fA-F0-9]{64}$ ]]; then
        printf 'ERROR: invalid image name: %s\n' "${name}" >&2
        exit 1
    fi
}

rx_met_image_exists() {
    docker image inspect "$1" >/dev/null 2>&1
}

rx_met_docker_build() {
    local -a args=("$@")
    if [[ -n "${RX_MET_BUILD_PROXY:-}" ]]; then
        args+=(
            --network host
            --build-arg "HTTP_PROXY=${RX_MET_BUILD_PROXY}"
            --build-arg "HTTPS_PROXY=${RX_MET_BUILD_PROXY}"
            --build-arg "http_proxy=${RX_MET_BUILD_PROXY}"
            --build-arg "https_proxy=${RX_MET_BUILD_PROXY}"
            --build-arg "NO_PROXY=localhost,127.0.0.1"
        )
    fi
    docker build "${args[@]}"
}

rx_met_require_image_name "${ADA200_IMAGE}"
rx_met_require_image_name "${DEV_IMAGE}"

for required in \
    "${ROOT}/docker/Dockerfile" \
    "${ROOT}/docker/Dockerfile.ada200" \
    "${ROOT}/requirements-build.txt"
do
    [[ -f "${required}" ]] || { log "ERROR: missing ${required}"; exit 1; }
done

BASE_IMAGE=""
if rx_met_image_exists "${ADA200_IMAGE}"; then
    BASE_IMAGE="${ADA200_IMAGE}"
    log "found official runtime ${BASE_IMAGE}"
elif rx_met_image_exists "ada200_docker"; then
    BASE_IMAGE="ada200_docker"
    log "found official runtime ${BASE_IMAGE}"
else
    log "ada200_docker not found; build runtime from docker/Dockerfile.ada200"
    log "prefer loading the official image when available:"
    log "  docker load -i /mnt/data2/reexen_release/ADA200/docker/v1.0.0/artifacts/ada200_docker_v1.tar.gz"
    iidfile="$(mktemp)"
    rx_met_docker_build \
        -f "${ROOT}/docker/Dockerfile.ada200" \
        --iidfile "${iidfile}" \
        "${ROOT}"
    BASE_IMAGE="$(tr -d '[:space:]' < "${iidfile}")"
    rm -f "${iidfile}"
    log "runtime image id ${BASE_IMAGE}"
fi
rx_met_require_image_name "${BASE_IMAGE}"

log "build ${DEV_IMAGE} FROM ${BASE_IMAGE}"
rx_met_docker_build \
    -f "${ROOT}/docker/Dockerfile" \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    -t "${DEV_IMAGE}" \
    "${ROOT}"

log "verify ${DEV_IMAGE}"
docker run --rm "${DEV_IMAGE}" python3 - <<'PY'
import platform
import shutil
import sys

if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 10):
    raise SystemExit(
        f"need CPython 3.10, found {sys.implementation.name} {sys.version_info.major}.{sys.version_info.minor}"
    )
if platform.system() != "Linux" or platform.machine() != "x86_64":
    raise SystemExit(f"need Linux x86_64, found {platform.system()} {platform.machine()}")

import torch

ver = torch.__version__.split("+", 1)[0]
if ver != "2.8.0":
    raise SystemExit(f"need torch==2.8.0, found {torch.__version__}")
if getattr(torch.version, "cuda", None) != "12.8":
    raise SystemExit(f"need torch.version.cuda == 12.8, found {torch.version.cuda}")
if shutil.which("nvcc") is None:
    raise SystemExit("nvcc is required in ada200_docker:rx-met-dev")
if shutil.which("cmake") is None:
    raise SystemExit("cmake is required in ada200_docker:rx-met-dev")

from pathlib import Path

cuda_includes = [
    Path("/usr/local/cuda/targets/x86_64-linux/include"),
    Path("/usr/local/cuda/include"),
]
for header in ("curand_kernel.h", "cudnn.h"):
    if not any((path / header).exists() for path in cuda_includes):
        raise SystemExit(f"missing {header} on the CUDA include path")
print("dev image ok", "python", sys.version.split()[0], "torch", torch.__version__)
PY

log "done: ${DEV_IMAGE}"
log "official customer runtime remains ada200_docker:latest"
log "next: ./scripts/release_build.sh"
