#!/usr/bin/env bash
# Shared helpers for the reex-gemm-datagen one-click scripts.
#
# Env overrides:
#   BUILD_DIR  cmake build dir under llama.cpp root      (default build_cuda_q64)
#   OUT        output root for generated products         (default <tool>/output)
#   M / N / K  GEMM shape override (else binary defaults: 8192 / 2048 / 2048)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LLAMA_ROOT="$(cd "$TOOL_DIR/../.." && pwd)"

BUILD_DIR="${BUILD_DIR:-build_cuda_q64}"
BIN="$LLAMA_ROOT/$BUILD_DIR/bin/reex-gemm-datagen"
OUT_DIR="${OUT:-$TOOL_DIR/output}"

# Optional shape override (export M / N / K to change the GEMM shape).
SHAPE_ARGS=()
[[ -n "${M:-}" ]] && SHAPE_ARGS+=(--M "$M")
[[ -n "${N:-}" ]] && SHAPE_ARGS+=(--N "$N")
[[ -n "${K:-}" ]] && SHAPE_ARGS+=(--K "$K")

require_bin() {
    if [[ ! -x "$BIN" ]]; then
        echo "error: binary not found at $BIN" >&2
        echo "       build it first:  $SCRIPT_DIR/build.sh" >&2
        exit 1
    fi
}

# Per-group sub-directory under OUT_DIR (set by each gen_*.sh: legacy / kquant / int).
GROUP="${GROUP:-}"
group_out() { printf '%s' "$OUT_DIR${GROUP:+/$GROUP}"; }

# run_case <datagen-args...>
run_case() {
    require_bin
    local dest; dest="$(group_out)"
    mkdir -p "$dest"
    echo "==> reex-gemm-datagen --out $dest $* ${SHAPE_ARGS[*]:-}"
    "$BIN" --out "$dest" "$@" ${SHAPE_ARGS[@]+"${SHAPE_ARGS[@]}"}
}

# run_quant_sweep <wtype> <wbits>
# Standard act_in -> A_bits sweep shared by the quantized groups (legacy / k-quant),
# matching the rgd_out reference coverage:
#   F32->A16, F16->A8, BF16->A8, E5M2->A4, E4M3->A4
# A4 (E5M2/E4M3) is a valid compute mode only with W<=4, so it is skipped for wider W.
run_quant_sweep() {
    local wt="$1" wbits="$2"
    run_case --wtype "$wt" --abits 16 --actin F32
    run_case --wtype "$wt" --abits 8  --actin F16
    run_case --wtype "$wt" --abits 8  --actin BF16
    if (( wbits <= 4 )); then
        run_case --wtype "$wt" --abits 4 --actin E5M2
        run_case --wtype "$wt" --abits 4 --actin E4M3
    fi
}
