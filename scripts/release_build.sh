#!/usr/bin/env bash
# Full offline release: verified Docker image + examples + customer README.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_FILE="${ROOT}/VERSION"
PRODUCT_VERSION="$(tr -d '[:space:]' < "${VERSION_FILE}")"
VERSION="${RX_MET_VERSION:-${PRODUCT_VERSION}}"
IMAGE="rx-met:${VERSION}"
EXPORT_DIR="${RX_MET_EXPORT_DIR:-${ROOT}/.release/export}"
RELEASE_NAME="rx-met-${VERSION}"
RELEASE_BUNDLE="${EXPORT_DIR}/${RELEASE_NAME}-release.tar.gz"
RELEASE_SHA256="${RELEASE_BUNDLE}.sha256"
README_FILE="${ROOT}/release/README.md"
STAGING_ROOT="${ROOT}/.release/staging"
GZIP_LEVEL="${RX_MET_GZIP_LEVEL:-6}"
CUDA_ARCHITECTURES="${RX_MET_CUDA_ARCHITECTURES:-80;86;89;120}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;12.0}"
STAGE_PARENT=""
TEMP_BUNDLE=""

log() { printf '[release_build] %s\n' "$*"; }
cleanup() {
    [[ -z "${STAGE_PARENT}" ]] || rm -rf "${STAGE_PARENT}"
    [[ -z "${TEMP_BUNDLE}" ]] || rm -f "${TEMP_BUNDLE}"
}
trap cleanup EXIT

if [[ ! "${PRODUCT_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.-]+)?$ ]]; then
    log "ERROR: invalid product version in ${VERSION_FILE}: ${PRODUCT_VERSION}"
    exit 1
fi
if [[ "${VERSION}" != "${PRODUCT_VERSION}" ]]; then
    log "ERROR: RX_MET_VERSION=${VERSION} differs from ${VERSION_FILE} (${PRODUCT_VERSION})"
    exit 1
fi
if [[ ! "${GZIP_LEVEL}" =~ ^[1-9]$ ]]; then
    log "ERROR: RX_MET_GZIP_LEVEL must be an integer from 1 to 9"
    exit 1
fi
if [[ ! -f "${README_FILE}" ]]; then
    log "ERROR: release README not found: ${README_FILE}"
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
    docker_args+=(
        --network host
        --build-arg "HTTP_PROXY=${RX_MET_BUILD_PROXY}"
        --build-arg "HTTPS_PROXY=${RX_MET_BUILD_PROXY}"
        --build-arg "http_proxy=${RX_MET_BUILD_PROXY}"
        --build-arg "https_proxy=${RX_MET_BUILD_PROXY}"
        --build-arg "NO_PROXY=localhost,127.0.0.1"
    )
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
tar -C "${ROOT}" \
    --exclude='examples/data' \
    --exclude='examples/output' \
    --exclude='*/__pycache__' \
    --exclude='*.pyc' \
    -cf - examples | tar -C "${RELEASE_DIR}" -xf -
cp "${README_FILE}" "${RELEASE_DIR}/README.md"
cp "${ROOT}/release/env.example" "${RELEASE_DIR}/env.example"
cp -a "${ROOT}/release/scripts" "${RELEASE_DIR}/scripts"
chmod +x "${RELEASE_DIR}/scripts/"*.sh
sed -i "s/@VERSION@/${VERSION}/g" \
    "${RELEASE_DIR}/README.md" \
    "${RELEASE_DIR}/env.example"
(
    cd "${RELEASE_DIR}"
    sha256sum "$(basename "${IMAGE_TAR}")" > SHA256SUMS
)

log "compress release bundle -> ${RELEASE_BUNDLE}"
TEMP_BUNDLE="${RELEASE_BUNDLE}.tmp.$$"
tar -C "${STAGE_PARENT}" -cf - "${RELEASE_NAME}" | gzip "-${GZIP_LEVEL}" > "${TEMP_BUNDLE}"
mv -f "${TEMP_BUNDLE}" "${RELEASE_BUNDLE}"
TEMP_BUNDLE=""
(
    cd "${EXPORT_DIR}"
    sha256sum "$(basename "${RELEASE_BUNDLE}")" > "$(basename "${RELEASE_SHA256}")"
)

# Remove artifacts produced by the previous image-only release layout.
rm -f \
    "${EXPORT_DIR}/${RELEASE_NAME}-image.tar.gz" \
    "${EXPORT_DIR}/${RELEASE_NAME}-image.tar.gz.sha256"

log "done:"
log "  image:   ${IMAGE}"
log "  release: ${RELEASE_BUNDLE} ($(du -h "${RELEASE_BUNDLE}" | cut -f1))"
log "  sha256:  ${RELEASE_SHA256}"
