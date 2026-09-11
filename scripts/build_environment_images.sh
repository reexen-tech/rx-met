#!/usr/bin/env bash
# 构建可跨多个 rx-met 产品版本复用的稳定 build-env/runtime-env。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BAKE_FILE="${ROOT}/docker/docker-bake.hcl"
source "${ROOT}/scripts/lib/environment_images.sh"
BUILDER="${RX_MET_BUILDER:-}"
FORCE=0
VERIFY=1
EXPORT=1
EXPORT_DIR=""
TARGETS=()

log() { printf '[build_environment_images] %s\n' "$*"; }
die() { printf '[build_environment_images] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
用法: ./scripts/build_environment_images.sh [选项] [cu118|cu126|cu130 ...]

选项：
  --force             即使对应环境镜像已存在也重新构建
  --no-verify         构建后不运行环境镜像验收
  --builder NAME      指定 Buildx builder
  --no-export         只准备本机环境镜像，不生成可搬运归档
  --export-dir DIR    构建后导出到指定本地目录
  -h, --help          显示帮助

默认处理全部三个 CUDA 变体，构建本机缺失的环境镜像，验证后生成本地归档。
代理使用当前 shell 的标准代理变量。
EOF
}

while (($#)); do
    case "$1" in
        --force) FORCE=1 ;;
        --no-verify) VERIFY=0 ;;
        --no-export) EXPORT=0 ;;
        --export-dir)
            (($# >= 2)) || die "--export-dir 缺少参数"
            EXPORT=1
            EXPORT_DIR="$2"
            shift
            ;;
        --builder)
            (($# >= 2)) || die "--builder 缺少参数"
            BUILDER="$2"
            shift
            ;;
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
for command in docker; do
    command -v "${command}" >/dev/null 2>&1 || die "缺少命令: ${command}"
done
docker info >/dev/null 2>&1 || die "无法访问 Docker daemon"
rx_met_require_docker_builder "${BUILDER}" || die "请切换到 default docker builder"

declare -A seen=()
targets_to_build=()
for target in "${TARGETS[@]}"; do
    [[ -z "${seen[${target}]:-}" ]] || die "目标重复: ${target}"
    seen["${target}"]=1
    build_image="$(rx_met_build_environment_image "${target}")"
    runtime_image="$(rx_met_runtime_environment_image "${target}")"
    build_exists=0
    runtime_exists=0
    docker image inspect "${build_image}" >/dev/null 2>&1 && build_exists=1
    docker image inspect "${runtime_image}" >/dev/null 2>&1 && runtime_exists=1

    if ((FORCE == 0 && build_exists && runtime_exists)); then
        log "复用已有环境镜像: ${target}"
        continue
    fi
    if ((FORCE == 0 && build_exists != runtime_exists)); then
        die "${target} 环境镜像不完整；请使用 --force 成对重建"
    fi
    targets_to_build+=("${target}")
done

export ENV_VERSION="${RX_MET_ENV_VERSION}"
export BUILD_ENV_REPOSITORY="${RX_MET_BUILD_ENV_REPOSITORY}"
export RUNTIME_ENV_REPOSITORY="${RX_MET_RUNTIME_ENV_REPOSITORY}"

if ((${#targets_to_build[@]})); then
    buildx_args=()
    if [[ -n "${BUILDER}" ]]; then
        buildx_args+=(--builder "${BUILDER}")
    fi
    proxy_count=0
    for proxy_var in HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY \
        http_proxy https_proxy all_proxy no_proxy; do
        proxy_value="${!proxy_var:-}"
        if [[ -n "${proxy_value}" ]]; then
            buildx_args+=(--set "*.args.${proxy_var}=${proxy_value}")
            ((proxy_count += 1))
        fi
    done
    ((proxy_count == 0)) || log "向本次构建传递 ${proxy_count} 个代理变量"
    for target in "${targets_to_build[@]}"; do
        log "构建环境: ${target}"
        (
            cd "${ROOT}"
            docker buildx bake \
                "${buildx_args[@]}" \
                -f "${BAKE_FILE}" \
                --load \
                --progress="${RX_MET_BUILD_PROGRESS:-plain}" \
                "environment-build-${target}" \
                "environment-runtime-${target}"
        )
    done
else
    log "所选环境镜像均已存在，无需构建"
fi

if ((VERIFY)); then
    "${ROOT}/scripts/verify_environment_images.sh" "${TARGETS[@]}"
fi
if ((EXPORT)); then
    export_args=()
    [[ -z "${EXPORT_DIR}" ]] || export_args+=(--output-dir "${EXPORT_DIR}")
    ((FORCE == 0)) || export_args+=(--force)
    "${ROOT}/scripts/export_environment_images.sh" \
        "${export_args[@]}" "${TARGETS[@]}"
fi
log "环境镜像准备完成"
