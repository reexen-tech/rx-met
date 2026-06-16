#!/usr/bin/env python3
"""Compare GPU(patched) vs CPU(ref) raw MoE dumps for the per-64 fixed-point check."""
import json, os, sys
import numpy as np

A = sys.argv[1] if len(sys.argv) > 1 else "/tmp/moe_gpu_dbg/case_00/raw"
B = sys.argv[2] if len(sys.argv) > 2 else "/tmp/moe_cpu_dbg/case_00/raw"
LAYER = int(sys.argv[3]) if len(sys.argv) > 3 else 0
GG = {"f32": np.float32, "f16": np.float16, "i32": np.int32}

def load(root, base, layer):
    j = json.load(open(os.path.join(root, f"{base}-{layer}.json")))
    arr = np.fromfile(os.path.join(root, f"{base}-{layer}.bin"), dtype=GG[j["ggml_type"]])
    ne = j["ne"]
    return arr.reshape(ne[3], ne[2], ne[1], ne[0])

def cmp(base):
    a = load(A, base, LAYER).astype(np.float64)
    b = load(B, base, LAYER).astype(np.float64)
    if a.shape != b.shape:
        print(f"  {base:24s} SHAPE MISMATCH gpu{a.shape} cpu{b.shape}"); return False
    if base == "ffn_moe_topk":
        eq = np.array_equal(a, b)
        print(f"  {base:24s} exact_equal={eq}")
        return eq
    diff = np.abs(a - b)
    denom = np.abs(b).max() + 1e-12
    amax, amean, rel = diff.max(), diff.mean(), diff.max()/denom
    ok = rel < 1e-4
    print(f"  {base:24s} abs_max={amax:.3e} abs_mean={amean:.3e} rel_max={rel:.3e}  {'OK' if ok else 'FAIL'}")
    return ok

print(f"GPU={A}\nCPU={B}\nlayer={LAYER}")

# Routing: GPU vs CPU diverge upstream (attention/softmax are float, not the
# per-64 fixed-point path), so some tokens flip experts. Isolate the per-64
# expert-GEMM datapath by comparing ONLY tokens whose top-k routing is identical.
tg = load(A, "ffn_moe_topk", LAYER)[0, 0]   # [T, K]
tc = load(B, "ffn_moe_topk", LAYER)[0, 0]
T, K = tg.shape
same = np.all(tg == tc, axis=1)             # exact same experts AND order
frac = same.mean()
print(f"\nrouting identical tokens: {same.sum()}/{T} ({frac*100:.2f}%)")

def cmp_masked(base, mask):
    a = load(A, base, LAYER)[0].astype(np.float64)  # [T, ...]
    b = load(B, base, LAYER)[0].astype(np.float64)
    a, b = a[mask], b[mask]
    diff = np.abs(a - b)
    denom = np.abs(b).max() + 1e-12
    print(f"  {base:24s} abs_max={diff.max():.3e} abs_mean={diff.mean():.3e} rel_max={diff.max()/denom:.3e}")
    return diff.max()/denom

print("\n[matched-routing tokens only] expert-GEMM datapath diff (GPU per-64 vs CPU per-64):")
for base in ["ffn_moe_down", "ffn_moe_weighted", "ffn_moe_out"]:
    cmp_masked(base, same)

# weights_norm uses [T,K] layout
def cmp_w(mask):
    a = load(A, "ffn_moe_weights_norm", LAYER)[0,0].astype(np.float64)[mask]
    b = load(B, "ffn_moe_weights_norm", LAYER)[0,0].astype(np.float64)[mask]
    diff = np.abs(a-b)
    print(f"  {'ffn_moe_weights_norm':24s} abs_max={diff.max():.3e} abs_mean={diff.mean():.3e}")
cmp_w(same)
print("\n(diff on matched tokens reflects upstream float divergence in the expert\n input activation, NOT the per-64 GEMM; near-zero would need identical inputs.)")
