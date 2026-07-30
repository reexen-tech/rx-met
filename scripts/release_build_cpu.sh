#!/usr/bin/env bash
# CPU-only offline release: verified Docker image + examples + customer README.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=release_common.sh
source "${ROOT}/scripts/release_common.sh"

VERSION_FILE="${ROOT}/VERSION"
VERSION="$(rx_met_require_version "${VERSION_FILE}")"
IMAGE="rx-met:${VERSION}-cpu"
EXPORT_DIR="${RX_MET_EXPORT_DIR:-${ROOT}/.release/export}"
RELEASE_NAME="rx-met-${VERSION}-cpu"
RELEASE_BUNDLE="${EXPORT_DIR}/${RELEASE_NAME}-release.tar.gz"
RELEASE_SHA256="${RELEASE_BUNDLE}.sha256"
README_SRC="${ROOT}/release/README.cpu.md"
ENV_SRC="${ROOT}/release/env.cpu.example"
STAGING_ROOT="${ROOT}/.release/staging"
GZIP_LEVEL="${RX_MET_GZIP_LEVEL:-6}"
STAGE_PARENT=""
TEMP_BUNDLE=""

log() { printf '[release_build_cpu] %s\n' "$*"; }
cleanup() {
    [[ -z "${STAGE_PARENT}" ]] || rm -rf "${STAGE_PARENT}"
    [[ -z "${TEMP_BUNDLE}" ]] || rm -f "${TEMP_BUNDLE}"
}
trap cleanup EXIT

if [[ ! "${GZIP_LEVEL}" =~ ^[1-9]$ ]]; then
    log "ERROR: RX_MET_GZIP_LEVEL must be an integer from 1 to 9"
    exit 1
fi
if [[ ! -f "${README_SRC}" ]]; then
    log "ERROR: CPU release README not found: ${README_SRC}"
    exit 1
fi
if [[ ! -f "${ENV_SRC}" ]]; then
    log "ERROR: CPU env example not found: ${ENV_SRC}"
    exit 1
fi
if [[ ! -f "${ROOT}/docker/Dockerfile.cpu" ]]; then
    log "ERROR: Dockerfile.cpu not found"
    exit 1
fi

export RX_MET_VERSION="${VERSION}"

docker_args=(
    --target runtime
    --build-arg "RX_MET_VERSION=${VERSION}"
    --build-arg "RX_MET_REVISION=$(git -C "${ROOT}" rev-parse HEAD)"
    -f "${ROOT}/docker/Dockerfile.cpu"
    -t "${IMAGE}"
)

if [[ -n "${RX_MET_BUILD_PROXY:-}" ]]; then
    log "build proxy enabled: ${RX_MET_BUILD_PROXY}"
    while IFS= read -r arg; do
        [[ -n "${arg}" ]] || continue
        docker_args+=("${arg}")
    done < <(rx_met_docker_proxy_args)
fi

log "multi-stage docker build -> ${IMAGE}"
log "note: quant-gru is skipped (CUDA-only); small-model quick_start.py is unavailable in this package"
DOCKER_BUILDKIT=1 docker build "${docker_args[@]}" "${ROOT}"

log "verify -> ${IMAGE}"
RX_MET_IMAGE="${IMAGE}" "${ROOT}/scripts/verify_image_cpu.sh"

mkdir -p "${EXPORT_DIR}" "${STAGING_ROOT}"
STAGE_PARENT="$(mktemp -d "${STAGING_ROOT}/${RELEASE_NAME}.XXXXXX")"
RELEASE_DIR="${STAGE_PARENT}/${RELEASE_NAME}"
IMAGE_TAR="${RELEASE_DIR}/${RELEASE_NAME}-image.tar"
mkdir -p "${RELEASE_DIR}"

log "docker save -> ${IMAGE_TAR}"
docker save -o "${IMAGE_TAR}" "${IMAGE}"

log "assemble examples + README + scripts"
rx_met_assemble_bundle_files

log "compress release bundle -> ${RELEASE_BUNDLE}"
rx_met_compress_release_bundle

log "done:"
log "  image:   ${IMAGE}"
log "  release: ${RELEASE_BUNDLE} ($(du -h "${RELEASE_BUNDLE}" | cut -f1))"
log "  sha256:  ${RELEASE_SHA256}"
log "  note:    QuantGRU / examples/quick_start.py not included (CUDA-only)"
