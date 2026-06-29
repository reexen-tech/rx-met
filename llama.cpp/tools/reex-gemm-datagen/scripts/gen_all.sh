#!/usr/bin/env bash
# Generate all three groups (legacy + k-quant + int) into the output dir.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/gen_legacy.sh"
"$SCRIPT_DIR/gen_kquant.sh"
"$SCRIPT_DIR/gen_int.sh"

echo "[all] done"
