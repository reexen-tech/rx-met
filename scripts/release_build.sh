#!/usr/bin/env bash
# Full offline release (GPU): verified Docker image + examples + customer README.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=release_common.sh
source "${ROOT}/scripts/release_common.sh"

VERSION_FILE="${ROOT}/VERSION"
VERSION="$(rx_met_require_version "${VERSION_FILE}")"
IMAGE="rx-met:${VERSION}"
EXPORT_DIR="${RX_MET_EXPORT_DIR:-${ROOT}/.release/export}"
RELEASE_NAME="rx-met-${VERSION}"
RELEASE_BUNDLE="${EXPORT_DIR}/${RELEASE_NAME}-release.tar.gz"
RELEASE_SHA256="${RELEASE_BUNDLE}.sha256"
README_SRC="${ROOT}/release/README.md"
ENV_SRC="${ROOT}/release/env.example"
STAGING_ROOT="${ROOT}/.release/staging"
GZIP_LEVEL="${RX_MET_GZIP_LEVEL:-6}"
CUDA_ARCHITECTURES="${RX_MET_CUDA_ARCHITECTURES:-80;86;89;90;120}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0;12.0}"
STAGE_PARENT=""
TEMP_BUNDLE=""

log() { printf '[release_build] %s\n' "$*"; }
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
    log "ERROR: release README not found: ${README_SRC}"
    exit 1
fi
export RX_MET_VERSION="${VERSION}"

docker_args=(
    --target runtime
    --build-arg "RX_MET_VERSION=${VERSION}"
    --build-arg "RX_MET_REVISION=$(git -C "${ROOT}" rev-parse HEAD)"
    --build-arg "CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}"
    --build-arg "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
    -f "${ROOT}/docker/Dockerfile"
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
DOCKER_BUILDKIT=1 docker build "${docker_args[@]}" "${ROOT}"

log "verify -> ${IMAGE}"
RX_MET_IMAGE="${IMAGE}" "${ROOT}/scripts/verify_image.sh"

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

# Remove artifacts produced by the previous image-only release layout.
rm -f \
    "${EXPORT_DIR}/${RELEASE_NAME}-image.tar.gz" \
    "${EXPORT_DIR}/${RELEASE_NAME}-image.tar.gz.sha256"

log "done:"
log "  image:   ${IMAGE}"
log "  release: ${RELEASE_BUNDLE} ($(du -h "${RELEASE_BUNDLE}" | cut -f1))"
log "  sha256:  ${RELEASE_SHA256}"
