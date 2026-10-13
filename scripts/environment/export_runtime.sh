#!/usr/bin/env bash
# Export runtime-only gzip archives; build environments remain internal.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python3 "$ROOT/scripts/release/artifacts.py" export-runtime "$@"
