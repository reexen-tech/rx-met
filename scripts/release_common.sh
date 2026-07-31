#!/usr/bin/env bash
# Shared helpers for GPU/CPU release_build scripts.
# shellcheck disable=SC2034

rx_met_require_version() {
    local version_file="$1"
    local product_version version
    product_version="$(tr -d '[:space:]' < "${version_file}")"
    version="${RX_MET_VERSION:-${product_version}}"
    if [[ ! "${product_version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.-]+)?$ ]]; then
        printf 'ERROR: invalid product version in %s: %s\n' \
            "${version_file}" "${product_version}" >&2
        exit 1
    fi
    if [[ "${version}" != "${product_version}" ]]; then
        printf 'ERROR: RX_MET_VERSION=%s differs from %s (%s)\n' \
            "${version}" "${version_file}" "${product_version}" >&2
        exit 1
    fi
    printf '%s\n' "${version}"
}

# Assemble examples/README/env/scripts into RELEASE_DIR and write SHA256SUMS
# for the already-exported IMAGE_TAR basename.
#
# Required variables:
#   ROOT, RELEASE_DIR, README_SRC, ENV_SRC, VERSION, IMAGE_TAR
rx_met_assemble_bundle_files() {
    tar -C "${ROOT}" \
        --exclude='examples/data' \
        --exclude='examples/output' \
        --exclude='*/__pycache__' \
        --exclude='*.pyc' \
        -cf - examples | tar -C "${RELEASE_DIR}" -xf -
    cp "${README_SRC}" "${RELEASE_DIR}/README.md"
    cp "${ENV_SRC}" "${RELEASE_DIR}/env.example"
    cp -a "${ROOT}/release/scripts" "${RELEASE_DIR}/scripts"
    chmod +x "${RELEASE_DIR}/scripts/"*.sh
    sed -i "s/@VERSION@/${VERSION}/g" \
        "${RELEASE_DIR}/README.md" \
        "${RELEASE_DIR}/env.example"
    (
        cd "${RELEASE_DIR}"
        sha256sum "$(basename "${IMAGE_TAR}")" > SHA256SUMS
    )
}

# Compress RELEASE_NAME under STAGE_PARENT into RELEASE_BUNDLE (+ .sha256).
#
# Required variables:
#   STAGE_PARENT, RELEASE_NAME, RELEASE_BUNDLE, RELEASE_SHA256, EXPORT_DIR, GZIP_LEVEL
rx_met_compress_release_bundle() {
    local temp_bundle="${RELEASE_BUNDLE}.tmp.$$"
    mkdir -p "${EXPORT_DIR}"
    tar -C "${STAGE_PARENT}" -cf - "${RELEASE_NAME}" | gzip "-${GZIP_LEVEL}" > "${temp_bundle}"
    mv -f "${temp_bundle}" "${RELEASE_BUNDLE}"
    (
        cd "${EXPORT_DIR}"
        sha256sum "$(basename "${RELEASE_BUNDLE}")" > "$(basename "${RELEASE_SHA256}")"
    )
}

rx_met_docker_proxy_args() {
    if [[ -z "${RX_MET_BUILD_PROXY:-}" ]]; then
        return 0
    fi
    printf '%s\n' \
        --network host \
        --build-arg "HTTP_PROXY=${RX_MET_BUILD_PROXY}" \
        --build-arg "HTTPS_PROXY=${RX_MET_BUILD_PROXY}" \
        --build-arg "http_proxy=${RX_MET_BUILD_PROXY}" \
        --build-arg "https_proxy=${RX_MET_BUILD_PROXY}" \
        --build-arg "NO_PROXY=localhost,127.0.0.1"
}
