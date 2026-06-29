#!/usr/bin/env bash
# Generate the Legacy-quant group products (q8_0_64 / q8_1_64s / q4_0_64).
#
# For each wtype runs the standard act_in -> A_bits sweep (run_quant_sweep):
#   W8 (q8_0_64 / q8_1_64s): A16-F32 / A8-F16 / A8-BF16            (3 cases)
#   W4 (q4_0_64):            + A4-E5M2 / A4-E4M3                    (5 cases)
#
# Env: BUILD_DIR/OUT/M/N/K overrides from _common.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUP=legacy
source "$SCRIPT_DIR/_common.sh"

run_quant_sweep q8_0_64  8
run_quant_sweep q8_1_64s 8
run_quant_sweep q4_0_64  4

echo "[legacy] done -> $(group_out)"
