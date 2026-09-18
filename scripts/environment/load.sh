#!/usr/bin/env bash
# 校验环境归档并将 build-env/runtime-env 成对加载到本机 Docker image store。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/scripts/lib/environment_images.sh"
ARCHIVE_DIR="${RX_MET_ENV_ARCHIVE_DIR:-}"
FORCE=0
VERIFY=1
TARGETS=()
EXPLICIT_ARCHIVES=()

log() { printf '[load_environment_images] %s\n' "$*"; }
die() { printf '[load_environment_images] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
用法: ./scripts/environment/load.sh [选项] [cu118|cu126|cu130 ...]

选项：
  --archive-dir DIR   按标准文件名从 DIR 加载所选变体
  --archive FILE      显式指定一个环境归档；可重复使用
  --force             允许覆盖本机已有的环境镜像 tag
  --no-verify         加载后不运行环境镜像验收
  -h, --help          显示帮助

归档所在目录必须包含 SHA256SUMS。默认加载全部三个 CUDA 变体。
EOF
}

while (($#)); do
    case "$1" in
        --archive-dir)
            (($# >= 2)) || die "--archive-dir 缺少参数"
            ARCHIVE_DIR="$2"
            shift
            ;;
        --archive)
            (($# >= 2)) || die "--archive 缺少参数"
            EXPLICIT_ARCHIVES+=("$2")
            shift
            ;;
        --force) FORCE=1 ;;
        --no-verify) VERIFY=0 ;;
        -h|--help) usage; exit 0 ;;
        cu118|cu126|cu130) TARGETS+=("$1") ;;
        *) die "未知参数: $1" ;;
    esac
    shift
done
if ((${#TARGETS[@]} == 0)); then
    TARGETS=(cu118 cu126 cu130)
fi
rx_met_validate_environment_config || die "环境镜像名称或版本无效"
for command in docker sha256sum zstd; do
    command -v "${command}" >/dev/null 2>&1 || die "缺少命令: ${command}"
done
docker info >/dev/null 2>&1 || die "无法访问 Docker daemon"
rx_met_require_docker_builder "${RX_MET_BUILDER:-}" \
    || die "请切换到 default docker builder"

declare -A explicit_by_target=()
for archive in "${EXPLICIT_ARCHIVES[@]}"; do
    [[ -f "${archive}" ]] || die "环境归档不存在: ${archive}"
    matched=""
    for target in cu118 cu126 cu130; do
        if [[ "$(basename "${archive}")" == "$(rx_met_environment_archive_name "${target}")" ]]; then
            matched="${target}"
            break
        fi
    done
    [[ -n "${matched}" ]] || die "无法从标准文件名识别 CUDA 变体: ${archive}"
    [[ -z "${explicit_by_target[${matched}]:-}" ]] \
        || die "重复指定 ${matched} 环境归档"
    explicit_by_target["${matched}"]="${archive}"
done

checksum_for() {
    local sums="$1"
    local filename="$2"
    local digest listed normalized found=""
    [[ -f "${sums}" ]] || die "缺少校验清单: ${sums}"
    while read -r digest listed; do
        normalized="${listed#\*}"
        normalized="${normalized#./}"
        if [[ "${normalized}" == "${filename}" ]]; then
            [[ -z "${found}" ]] || die "SHA256SUMS 中存在重复条目: ${filename}"
            found="${digest}"
        fi
    done < "${sums}"
    [[ "${found}" =~ ^[0-9a-f]{64}$ ]] \
        || die "SHA256SUMS 中缺少有效条目: ${filename}"
    printf '%s' "${found}"
}

declare -A seen=()
for target in "${TARGETS[@]}"; do
    [[ -z "${seen[${target}]:-}" ]] || die "目标重复: ${target}"
    seen["${target}"]=1
    archive="${explicit_by_target[${target}]:-}"
    if [[ -z "${archive}" ]]; then
        [[ -n "${ARCHIVE_DIR}" ]] \
            || die "${target} 未指定 --archive，且没有 --archive-dir"
        archive="${ARCHIVE_DIR}/$(rx_met_environment_archive_name "${target}")"
    fi
    [[ -f "${archive}" ]] || die "环境归档不存在: ${archive}"
    archive_dir="$(cd "$(dirname "${archive}")" && pwd)"
    archive_name="$(basename "${archive}")"
    expected="$(checksum_for "${archive_dir}/SHA256SUMS" "${archive_name}")"
    actual="$(sha256sum "${archive}")"
    actual="${actual%% *}"
    [[ "${actual}" == "${expected}" ]] \
        || die "环境归档 SHA-256 不匹配: ${archive}"

    build_image="$(rx_met_build_environment_image "${target}")"
    runtime_image="$(rx_met_runtime_environment_image "${target}")"
    if ((FORCE == 0)); then
        for image in "${build_image}" "${runtime_image}"; do
            ! docker image inspect "${image}" >/dev/null 2>&1 \
                || die "本机已存在 ${image}；拒绝覆盖，请先复用或显式使用 --force"
        done
    fi
    log "加载 ${archive_name}"
    zstd -dc "${archive}" | docker load
done

if ((VERIFY)); then
    "${ROOT}/scripts/environment/verify.sh" "${TARGETS[@]}"
fi
log "环境归档加载完成"
