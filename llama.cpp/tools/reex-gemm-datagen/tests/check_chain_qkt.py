#!/usr/bin/env python3
"""Independent NumPy recompute of the qkt chain dump (docs/chain-qkt.md).

Checks the whole two-stage pipeline from the dumped artifacts alone:
  1. decode mma1_act_blocks + mma1_weight_blocks -> integer MAC ->
     compare against mma1_rslt_f32 (tolerance: fp32 vs fp64 fold).
  2. re-quantize mma1_rslt_f32 (float32 amax -> scale -> RNE -> clamp,
     same math as the fused output-stage kernel) -> compare
     kblocks_pretrans BYTE-FOR-BYTE.
  3. block-atomic key<->head_dim grid transpose of pretrans ->
     compare kblocks_posttrans byte-for-byte.
  4. decode kblocks_posttrans + mma2_act_blocks -> integer MAC ->
     compare against mma2_golden_f32 and mma2_output_F16.

Usage: check_chain_qkt.py <case-dir>
"""
import json
import sys

import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "."
meta = json.load(open(f"{D}/meta.json"))

m1 = meta["mma1"]
m2 = meta["mma2"]
kb = meta["kblocks"]
M1, N1, K1 = m1["M"], m1["N"], m1["K"]
M2, N2, K2 = m2["M"], m2["N"], m2["K"]
A_BITS = m1["act_bits"]
KBITS = kb["kbits"]
KBB = kb["block_bytes"]
NUM_KEYS = meta["chain"]["num_keys"]
N_HD_GROUPS = meta["chain"]["head_dim"] // 64
ACT_DT = m1["act_input_dtype"]

fails = []


def check(name, ok, detail=""):
    print(f"  [{'OK' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        fails.append(name)


# ---- tile-order helpers (mirror reex_layout.h) ------------------------------
def wslot(n, sb, N, ts):
    Ntiles = N // ts["Nt"]
    return (sb * Ntiles + n // ts["Nt"]) * ts["Nt"] + n % ts["Nt"]


def aslot(m, kg, K, ts):                      # agroup == Kt (gpt == 1)
    Ktiles = K // ts["Kt"]
    return ((m // ts["Mt"]) * Ktiles + kg) * ts["Mt"] + m % ts["Mt"]


def rslot(m, n, M, ts):
    Mtiles = M // ts["Mt"]
    t = (n // ts["Nt"]) * Mtiles + m // ts["Mt"]
    return t * (ts["Mt"] * ts["Nt"]) + (m % ts["Mt"]) * ts["Nt"] + n % ts["Nt"]


# ---- block decoders ----------------------------------------------------------
def decode_legacy_wblocks(path, wtype, N, K, ts):
    """-> (d[N, K//64] f32, q[N, K] int32) from Legacy {d; qs} blocks."""
    if wtype == "q8_0_64":
        rec = np.dtype([("d", "<f2"), ("qs", "i1", 64)])
    elif wtype == "q8_1_64":
        rec = np.dtype([("d", "<f2"), ("s", "<f2"), ("qs", "i1", 64)])
    elif wtype == "q4_0_64":
        rec = np.dtype([("d", "<f2"), ("qs", "u1", 32)])
    else:
        raise SystemExit(f"unsupported wtype {wtype}")
    blk = np.fromfile(path, dtype=rec)
    d = np.zeros((N, K // 64), np.float32)
    q = np.zeros((N, K), np.int32)
    for n in range(N):
        for sb in range(K // 64):
            b = blk[wslot(n, sb, N, ts)]
            d[n, sb] = np.float32(b["d"])
            if wtype == "q4_0_64":                       # nibble interleave
                lo = (b["qs"] & 0x0F).astype(np.int32)
                hi = (b["qs"] >> 4).astype(np.int32)
                lo -= (lo >= 8) * 16                     # sign-extend signed 4-bit
                hi -= (hi >= 8) * 16
                q[n, sb * 64:sb * 64 + 32] = lo
                q[n, sb * 64 + 32:sb * 64 + 64] = hi
            else:
                q[n, sb * 64:(sb + 1) * 64] = b["qs"].astype(np.int32)
    return d, q


def decode_act_blocks(path, M, K, ts, a_bits):
    """-> (d[M, K//64] f32, q[M, K] int32) from { f16 d; intX qs[64] } groups."""
    ct = "<i2" if a_bits > 8 else "i1"
    rec = np.dtype([("d", "<f2"), ("qs", ct, 64)])
    blk = np.fromfile(path, dtype=rec)
    d = np.zeros((M, K // 64), np.float32)
    q = np.zeros((M, K), np.int32)
    for m in range(M):
        for kg in range(K // 64):
            b = blk[aslot(m, kg, K, ts)]
            d[m, kg] = np.float32(b["d"])
            q[m, kg * 64:(kg + 1) * 64] = b["qs"].astype(np.int32)
    return d, q


def int_gemm(wd, wq, ad, aq):
    """C[m,n] = sum_g wd[n,g]*ad[m,g]*(int dot of group g) — float64 fold."""
    M, N, G = aq.shape[0], wq.shape[0], wd.shape[1]
    C = np.zeros((M, N), np.float64)
    for g in range(G):
        sumi = aq[:, g * 64:(g + 1) * 64].astype(np.int64) @ \
               wq[:, g * 64:(g + 1) * 64].astype(np.int64).T
        C += ad[:, g:g + 1].astype(np.float64) * wd[:, g].astype(np.float64)[None, :] * sumi
    return C


# ---- 1) MMA1: blocks -> integer MAC -> vs mma1_rslt_f32 ---------------------
print(f"case {meta['name']}  MMA1[{M1},{N1},{K1}] -> K({kb['ktype']}) -> MMA2[{M2},{N2},{K2}]")
ts1, ts2 = m1["tiling"], m2["tiling"]

w1d, w1q = decode_legacy_wblocks(f"{D}/mma1_weight_blocks.bin", m1["weight_type"], N1, K1, ts1)
a1d, a1q = decode_act_blocks(f"{D}/mma1_act_blocks.bin", M1, K1, ts1, A_BITS)
C1_np = int_gemm(w1d, w1q, a1d, a1q)

rslt = np.fromfile(f"{D}/mma1_rslt_f32.bin", dtype=np.float32)
C1 = np.empty((M1, N1), np.float32)
for m in range(M1):
    for n in range(N1):
        C1[m, n] = rslt[rslot(m, n, M1, ts1)]
err = np.abs(C1_np - C1.astype(np.float64))
den = np.maximum(np.abs(C1.astype(np.float64)), 1e-6)
check("mma1 integer MAC vs mma1_rslt_f32", (err / den).max() < 1e-4,
      f"max_abs={err.max():.3e} max_rel={(err/den).max():.3e}")

# ---- 2) fused output quant recompute -> vs kblocks_pretrans (byte-exact) ----
QMAX = (1 << (KBITS - 1)) - 1
pre = np.fromfile(f"{D}/kblocks_pretrans.bin", dtype=np.uint8).reshape(-1, KBB)
mine = np.zeros_like(pre)
for key in range(NUM_KEYS):
    for g in range(N_HD_GROUPS):
        v = C1[key, g * 64:(g + 1) * 64]                     # float32 accumulator row
        amax = np.float32(np.abs(v).max())
        scale = amax / np.float32(QMAX) if amax > 0 else np.float32(1.0)
        inv = np.float32(QMAX) / amax if amax > 0 else np.float32(0.0)
        q = np.rint((v * inv).astype(np.float64)).astype(np.int64)   # RNE
        q = np.clip(q, -QMAX, QMAX)
        blk = mine[key * N_HD_GROUPS + g]
        blk[0:2] = np.array([scale], np.float16).view(np.uint8)
        if KBITS == 8:
            blk[2:66] = q.astype(np.int8).view(np.uint8)
        else:
            c4 = (q.astype(np.int64) & 0xF).astype(np.uint8)
            blk[2:34] = c4[:32] | (c4[32:] << 4)
nbad = int((mine != pre).sum())
check("fused quant recompute vs kblocks_pretrans (byte-exact)", nbad == 0,
      f"mismatched_bytes={nbad}/{pre.size}")

# ---- 3) block-atomic transpose -> vs kblocks_posttrans ----------------------
post = np.fromfile(f"{D}/kblocks_posttrans.bin", dtype=np.uint8).reshape(-1, KBB)
mine_post = np.zeros_like(post)
for key in range(NUM_KEYS):
    for g in range(N_HD_GROUPS):
        mine_post[g * NUM_KEYS + key] = pre[key * N_HD_GROUPS + g]
check("block transpose vs kblocks_posttrans (byte-exact)",
      bool((mine_post == post).all()))

# ---- 4) MMA2: posttrans + Q blocks -> integer MAC -> vs golden/output -------
kd, kq = decode_legacy_wblocks(f"{D}/kblocks_posttrans.bin", kb["ktype"], N2, K2, ts2)
qd, qq = decode_act_blocks(f"{D}/mma2_act_blocks.bin", M2, K2, ts2, A_BITS)
C2_np = int_gemm(kd, kq, qd, qq)

gold2 = np.fromfile(f"{D}/mma2_golden_f32.bin", dtype=np.float32)
G2 = np.empty((M2, N2), np.float32)
for m in range(M2):
    for n in range(N2):
        G2[m, n] = gold2[rslot(m, n, M2, ts2)]
err2 = np.abs(C2_np - G2.astype(np.float64))
den2 = np.maximum(np.abs(G2.astype(np.float64)), 1e-6)
check("mma2 integer MAC vs mma2_golden_f32", (err2 / den2).max() < 1e-4,
      f"max_abs={err2.max():.3e} max_rel={(err2/den2).max():.3e}")

of16 = np.fromfile(f"{D}/mma2_output_F16.bin", dtype=np.float16)
O2 = np.empty((M2, N2), np.float32)
for m in range(M2):
    for n in range(N2):
        O2[m, n] = np.float32(of16[rslot(m, n, M2, ts2)])
err3 = np.abs(C2_np - O2.astype(np.float64))
check("mma2 integer MAC vs mma2_output_F16", (err3 / den2).max() < 1e-2,
      f"max_abs={err3.max():.3e} (fp16 rounding expected)")

print("PASS" if not fails else f"FAIL: {fails}")
sys.exit(0 if not fails else 1)
