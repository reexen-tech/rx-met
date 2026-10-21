#!/usr/bin/env bash
# Maintainer-side verification helper; never shipped in application bundles.
INSTALL_PREFIX="${RX_MET_INSTALL_PREFIX:-${ROOT}/.release/installed}"
rx_met_resolve_installation() {
    local variant="$1"
    local version
    version="$(tr -d '[:space:]' < "$ROOT/VERSION")"
    RX_MET_INSTALLED_DIR="$(realpath -e -- "$INSTALL_PREFIX/rx-met-v${version}-${variant}-linux-amd64")" || return 1
    RX_MET_RUNTIME_ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime"]["image_id"])' "$RX_MET_INSTALLED_DIR/manifest.json")" || return 1
    docker image inspect "$RX_MET_RUNTIME_ID" >/dev/null || return 1
}
