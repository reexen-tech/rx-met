#!/usr/bin/env bash
# Generate the IntBlock (pure-integer, no-scale) group products — full sweep,
# matching the rgd_out reference coverage:
#   A bits in {16,8} x both signs x W bits {2,3,4,5,6,8} x both signs   (48 cases)
#   A bits  4        x both signs x W bits {2,3,4}       x both signs   (12 cases)
# (asign/wsign: i = signed two's-comp, u = unsigned)
#
# Env: BUILD_DIR/OUT/M/N/K overrides from _common.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUP=int
source "$SCRIPT_DIR/_common.sh"

# A_bits -> allowed W_bits (A4 only pairs with W<=4).
declare -A WBITS_FOR=( [16]="2 3 4 5 6 8" [8]="2 3 4 5 6 8" [4]="2 3 4" )

for ab in 16 8 4; do
    for as in i u; do
        for wb in ${WBITS_FOR[$ab]}; do
            for ws in i u; do
                run_case --wtype INT --abits "$ab" --asign "$as" --wbits "$wb" --wsign "$ws"
            done
        done
    done
done

echo "[int] done -> $(group_out)"
