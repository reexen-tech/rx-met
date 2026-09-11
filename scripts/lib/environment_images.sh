#!/usr/bin/env bash

RX_MET_ENV_VERSION="${RX_MET_ENV_VERSION:-deps-v1}"
RX_MET_BUILD_ENV_REPOSITORY="${RX_MET_BUILD_ENV_REPOSITORY:-rx-met-build-env}"
RX_MET_RUNTIME_ENV_REPOSITORY="${RX_MET_RUNTIME_ENV_REPOSITORY:-rx-met-runtime-env}"

rx_met_validate_environment_config() {
    [[ "${RX_MET_ENV_VERSION}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || return 1
    [[ "${RX_MET_BUILD_ENV_REPOSITORY}" =~ ^[A-Za-z0-9._/-]+$ ]] || return 1
    [[ "${RX_MET_RUNTIME_ENV_REPOSITORY}" =~ ^[A-Za-z0-9._/-]+$ ]] || return 1
}

rx_met_build_environment_image() {
    printf '%s:%s-%s' \
        "${RX_MET_BUILD_ENV_REPOSITORY}" "${RX_MET_ENV_VERSION}" "$1"
}

rx_met_runtime_environment_image() {
    printf '%s:%s-%s' \
        "${RX_MET_RUNTIME_ENV_REPOSITORY}" "${RX_MET_ENV_VERSION}" "$1"
}

rx_met_environment_archive_name() {
    printf 'rx-met-environment-%s-%s-linux-amd64.tar.zst' \
        "${RX_MET_ENV_VERSION}" "$1"
}

rx_met_require_docker_builder() {
    local builder="${1:-}"
    local inspect=(docker buildx inspect)
    local driver

    if [[ -n "${builder}" ]]; then
        inspect+=("${builder}")
    fi
    driver="$("${inspect[@]}" | sed -n 's/^Driver:[[:space:]]*//p' | head -n 1)"
    [[ "${driver}" == "docker" ]] || {
        printf '环境镜像要求 Buildx 使用 docker driver，当前为 %s\n' \
            "${driver:-unknown}" >&2
        return 1
    }
}
