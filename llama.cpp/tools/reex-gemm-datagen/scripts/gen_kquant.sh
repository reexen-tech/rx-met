#!/usr/bin/env bash
# Generate the K-quant group products (W6/W5/W4/W3/W2 block-64 super-blocks).
#
# For each wtype runs the standard act_in -> A_bits sweep (run_quant_sweep):
#   W>4 (Q6/Q5): A16-F32 / A8-F16 / A8-BF16                        (3 cases)
#   W<=4 (Q4/Q3/Q2): + A4-E5M2 / A4-E4M3                           (5 cases)
#
# Env: BUILD_DIR/OUT/M/N/K overrides from _common.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUP=kquant
source "$SCRIPT_DIR/_common.sh"

run_quant_sweep Q6_K_64  6
run_quant_sweep Q5_K_64S 5
run_quant_sweep Q4_K_64S 4
run_quant_sweep Q3_K_64  3
run_quant_sweep Q2_K_64S 2

echo "[k-quant] done -> $(group_out)"
