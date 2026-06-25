#!/usr/bin/env python3
"""Independent NumPy check of the Q5_K_64S HW dump layout.

Decodes weight_blocks.bin (HW layout: glb_scale + 4*[sub_scale + 64 codes@5bit])
and act_blocks.bin (contiguous group {f16 d; i8 qs[agroup]}), reconstructs floats,
matmuls a few rows and compares against golden_f32.bin (result-tiled order)."""
import sys, json, numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "."
meta = json.load(open(f"{D}/meta.json"))
M, N, K = (meta["gemm"][k] for k in ("M", "N", "K"))
Mt, Nt, Kt = (meta["tiling"][k] for k in ("Mt", "Nt", "Kt"))
AG = meta["tiling"]["act_group_elems"]      # agroup (== Kt)
ROWS = 64                       # rows of A to validate

Ntiles, Ktiles = N // Nt, K // Kt

def wslot(n, sb):
    nt, c = divmod(n, Nt)
    return (sb * Ntiles + nt) * Nt + c

def aslot(m, kg):                            # contiguous group, agroup == Kt
    return (m // Mt * Ktiles + kg) * Mt + m % Mt

def rslot(m, n):
    Mtiles = M // Mt
    nt, c = divmod(n, Nt)
    return (nt * Mtiles + m) * (Mt * Nt) + (m % Mt) * Nt + c

# ---- decode weights: LSB-first bitstream per block ----
#   glb(16b) + 4*[ sub_scale(6b signed) + 64*5bit codes ]   -> 1320 bits = 165 B
SCALE_BITS, W_BITS = 6, 5
wbb = (16 + 4 * (SCALE_BITS + 64 * W_BITS) + 7) // 8          # 165
wb = np.fromfile(f"{D}/weight_blocks.bin", dtype=np.uint8).reshape(-1, wbb)
W = np.zeros((N, K), np.float32)
pw = 1 << np.arange(W_BITS)
ps = 1 << np.arange(SCALE_BITS)
for n in range(N):
    for sb in range(K // Kt):
        bits = np.unpackbits(wb[wslot(n, sb)], bitorder="little")
        d = np.uint16(bits[0:16].dot(1 << np.arange(16))).view(np.float16).astype(np.float32)
        pos = 16
        for s in range(4):
            sc = int(bits[pos:pos + SCALE_BITS].dot(ps)); pos += SCALE_BITS
            if sc >= (1 << (SCALE_BITS - 1)): sc -= (1 << SCALE_BITS)   # sign-extend
            idx = pos + np.arange(64)[:, None] * W_BITS + np.arange(W_BITS)[None, :]
            codes = bits[idx].dot(pw).astype(np.int32); pos += 64 * W_BITS
            W[n, sb * Kt + s * 64 : sb * Kt + s * 64 + 64] = d * float(sc) * (codes - 16)

# ---- decode activations (contiguous group: d f16, qs[AG] i8) for ROWS rows ----
astride = 2 + AG
ab = np.fromfile(f"{D}/act_blocks.bin", dtype=np.uint8).reshape(-1, astride)
A = np.zeros((ROWS, K), np.float32)
for m in range(ROWS):
    for kg in range(K // AG):
        blk = ab[aslot(m, kg)]
        ad = blk[0:2].view(np.float16)[0].astype(np.float32)
        qs = blk[2:2 + AG].view(np.int8).astype(np.float32)
        A[m, kg * AG:(kg + 1) * AG] = ad * qs

# ---- validate FP16 source tiling: de-tile src, compare to dequant blocks ----
#      (wrong tiling formula -> O(1) random error; correct -> only quant error)
asrc = np.fromfile(f"{D}/act_src_f16.bin", dtype=np.float16).astype(np.float32)
Asrc = np.zeros((ROWS, K), np.float32)
for m in range(ROWS):
    for kg in range(K // AG):
        b = aslot(m, kg) * AG
        Asrc[m, kg * AG:(kg + 1) * AG] = asrc[b:b + AG]
print(f"act_src tiling:    max|src-dequant|={np.abs(Asrc - A).max():.3e}  (expect ~quant step)")

wsrc = np.fromfile(f"{D}/weight_src_f16.bin", dtype=np.float16).astype(np.float32)
Wsrc = np.zeros((N, K), np.float32)
for n in range(N):
    for sb in range(K // Kt):
        b = wslot(n, sb) * Kt
        Wsrc[n, sb * Kt:(sb + 1) * Kt] = wsrc[b:b + Kt]
print(f"weight_src tiling: max|src-dequant|={np.abs(Wsrc - W).max():.3e}  (expect ~quant step)")

C = A @ W.T                                   # [ROWS, N]

# ---- compare against golden (result-tiled) ----
gold = np.fromfile(f"{D}/golden_f32.bin", dtype=np.float32)
G = np.empty((ROWS, N), np.float32)
for m in range(ROWS):
    for n in range(N):
        G[m, n] = gold[rslot(m, n)]

err = np.abs(C - G)
den = np.maximum(np.abs(G), 1e-6)
print(f"rows={ROWS} N={N}  max_abs={err.max():.3e}  "
      f"max_rel={(err/den).max():.3e}  mean_abs={err.mean():.3e}")
print("PASS" if err.max() < 1e-2 else "FAIL")
