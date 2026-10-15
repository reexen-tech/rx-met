#!/usr/bin/env bash
# Build three small application bundles against already published environment images.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python3 "$ROOT/scripts/release/artifacts.py" build "$@"
