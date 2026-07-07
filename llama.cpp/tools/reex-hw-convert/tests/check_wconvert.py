#!/usr/bin/env python3
"""Byte-level check of reex-hw-convert output (Legacy block-64).

The Legacy conversion is a pure WHOLE-BLOCK reorder into §4.1 tile order — the
reex struct bytes inside each block are untouched. So for every (n, sb) the block
at weight_blocks.bin[weight_block_slot(n,sb)] must be byte-identical to the block
at weight_native.bin[n*(K/Kt)+sb]. Type-agnostic: covers all 6 legacy types.

Usage:  check_wconvert.py <out-dir>
"""
import sys, json, numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "."
meta = json.load(open(f"{D}/meta.json"))

N, K   = meta["gemm"]["N"], meta["gemm"]["K"]
Nt, Kt = meta["tiling"]["Nt"], meta["tiling"]["Kt"]
w      = meta["weight"]
bb     = w["block_bytes"]          # bytes per block in weight_blocks.bin
nbb    = w["native_block_bytes"]   # bytes per block in weight_native.bin

if w["repacked"]:
    print("repacked (K-quant) output — byte-level reorder check not applicable; skipping.")
    sys.exit(0)

assert bb == nbb, f"legacy block_bytes {bb} != native_block_bytes {nbb}"

Ntiles = N // Nt
sb_per_row = K // Kt

def wslot(n, sb):
    return (sb * Ntiles + n // Nt) * Nt + n % Nt

blocks = np.fromfile(f"{D}/weight_blocks.bin", dtype=np.uint8).reshape(-1, bb)
native = np.fromfile(f"{D}/weight_native.bin", dtype=np.uint8).reshape(-1, nbb)

exp_blocks = N * sb_per_row
assert blocks.shape[0] == exp_blocks, f"weight_blocks n_blocks {blocks.shape[0]} != {exp_blocks}"
assert native.shape[0] == exp_blocks, f"weight_native n_blocks {native.shape[0]} != {exp_blocks}"

mism = 0
for n in range(N):
    for sb in range(sb_per_row):
        src = native[n * sb_per_row + sb]
        dst = blocks[wslot(n, sb)]
        if not np.array_equal(src, dst):
            mism += 1
            if mism <= 5:
                print(f"  mismatch at n={n} sb={sb} (slot {wslot(n,sb)})")

print(f"wtype={w['wtype']}  N={N} K={K}  blocks={exp_blocks}  block_bytes={bb}  mismatches={mism}")
print("PASS" if mism == 0 else "FAIL")
sys.exit(0 if mism == 0 else 1)
