#!/usr/bin/env python3
"""mmid_postprocess.py — finalize the REAL expert-reorder capture for RTL
"专家重排" verification.

Input: output/case_XX/mmid_raw/<op>-<L>.{router,ids_dst,expert_bounds}.bin,
captured live during a real prefill by the REEX_DUMP_MMID_DIR hook in
ggml_cuda_mul_mat_id. For every block-64 MoE mul_mat_id we captured, straight
from the live GPU tensors:
  router.bin        the routing table `ids`            [n_tokens, n_expert_used] int32
  ids_dst.bin       ggml mm_ids_helper output          [n_tokens*n_expert_used] int32
  expert_bounds.bin per-expert segment offsets         [n_experts+1] int32

Everything below is REAL captured data (router + reorder come from the SAME
live tensors of the SAME run); we do NOT derive the reorder ourselves.
token_slot_to_row is simply the inverse permutation of the captured ids_dst
(a re-indexing of real data, not a re-computation of the reorder).

Outputs, written under output/case_XX/layer_LL/ (overwriting the old, possibly
stale, routing artifacts so the whole case is internally consistent):
  router_expert_ids.npy        [T, K] int32   router[t, slot] = expert id
  expert_row_to_token_slot.npy [T*K]  int32   expert-major row r -> token*K + slot
  expert_bounds.npy            [E+1]  int32   expert e owns rows [bounds[e], bounds[e+1])
  token_slot_to_row.npy        [T, K] int32   (t, slot) -> expert-major row r
  expert_token_map.json        {expert: [{token_id, topk_slot}, ...]} (token-asc)
  expert_load_ratio.json       {n_expert, n_tokens, topk, total_assignments, experts}

Validation is purely INTERNAL self-consistency of the real capture (the captured
router/reorder must agree with each other). A non-fatal report of how much the
fresh routing differs from the previous router_expert_ids.npy is also printed.
"""
import json
import os
import sys

import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else \
    "/mnt/data8t/zcx/aimet_rx/llama.cpp/output/case_00"
MMID_DIR = os.path.join(ROOT, "mmid_raw")
OPS = ["ffn_moe_down", "ffn_moe_gate", "ffn_moe_up"]


def read_i32(path):
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.int32).copy()


def discover_layers():
    layers = set()
    for fn in os.listdir(MMID_DIR):
        if fn.endswith(".ids_dst.bin"):
            base = fn[:-len(".ids_dst.bin")]
            if "-" in base:
                layers.add(int(base.rsplit("-", 1)[1]))
    return sorted(layers)


def load_op(op, L):
    """Return (router[T,K], ids_dst[T*K], bounds[E+1], K, E) or None."""
    p = os.path.join(MMID_DIR, f"{op}-{L}")
    if not os.path.exists(p + ".ids_dst.bin"):
        return None
    meta = json.load(open(p + ".mmid.json"))
    K = int(meta["n_expert_used"])
    E = int(meta["n_experts"])
    router_flat = read_i32(p + ".router.bin")
    ids_dst = read_i32(p + ".ids_dst.bin")
    bounds = read_i32(p + ".expert_bounds.bin")
    T = router_flat.size // K
    return router_flat.reshape(T, K), ids_dst, bounds, K, E


def main():
    if not os.path.isdir(MMID_DIR):
        print(f"ERROR: {MMID_DIR} not found")
        return 1
    layers = discover_layers()
    if not layers:
        print(f"ERROR: no *-<L>.ids_dst.bin under {MMID_DIR}")
        return 1

    fails = 0
    div_tokens_total = 0
    div_tokens_count = 0
    for L in layers:
        # --- load + check the 3 ops captured for this layer agree ---
        ref = None
        ops_found = []
        for op in OPS:
            got = load_op(op, L)
            if got is None:
                continue
            ops_found.append(op)
            if ref is None:
                ref = got
            else:
                if not (np.array_equal(got[0], ref[0]) and
                        np.array_equal(got[1], ref[1]) and
                        np.array_equal(got[2], ref[2])):
                    print(f"L{L:02d} FAIL: op {op} disagrees with {ops_found[0]}")
                    fails += 1
        router, ids_dst, bounds, K, E = ref
        T = router.shape[0]
        n_rows = T * K

        # --- internal self-consistency of the REAL capture ---
        flat = router.reshape(-1)
        is_perm = (ids_dst.size == n_rows and
                   np.array_equal(np.sort(ids_dst), np.arange(n_rows)))
        # inverse permutation: (token, slot) -> expert-major row
        ts2row = np.empty(n_rows, dtype=np.int32)
        ts2row[ids_dst] = np.arange(n_rows, dtype=np.int32)
        ok_inverse = np.array_equal(ids_dst[ts2row], np.arange(n_rows))
        # expert owning each row (from bounds) must equal the captured router id
        expert_of_row = (np.searchsorted(
            bounds, np.arange(n_rows), side="right") - 1).astype(np.int32)
        ok_expert = np.array_equal(flat[ids_dst].astype(np.int32), expert_of_row)
        # per-expert counts from bounds match the router histogram
        counts = np.diff(bounds)
        ok_counts = np.array_equal(counts, np.bincount(flat, minlength=E).astype(np.int32))
        # within each expert, rows are in ascending token order
        ok_order = True
        for e in range(E):
            seg = ids_dst[int(bounds[e]):int(bounds[e + 1])]
            if seg.size > 1 and np.any(np.diff(seg // K) < 0):
                ok_order = False
                break

        all_ok = is_perm and ok_inverse and ok_expert and ok_counts and ok_order
        if not all_ok:
            fails += 1

        # --- informational: divergence from the previous router (if any) ---
        ldir = os.path.join(ROOT, f"layer_{L:02d}")
        old_path = os.path.join(ldir, "router_expert_ids.npy")
        div_note = ""
        if os.path.exists(old_path):
            try:
                old = np.load(old_path)
                if old.shape == router.shape:
                    exact = int(np.all(old == router, axis=1).sum())
                    div_tokens_total += T
                    div_tokens_count += (T - exact)
                    div_note = f" oldmatch={exact}/{T}"
            except Exception:
                pass

        # --- write real-capture products (overwrite stale routing artifacts) ---
        os.makedirs(ldir, exist_ok=True)
        np.save(os.path.join(ldir, "router_expert_ids.npy"), router.astype(np.int32))
        np.save(os.path.join(ldir, "expert_row_to_token_slot.npy"), ids_dst.astype(np.int32))
        np.save(os.path.join(ldir, "expert_bounds.npy"), bounds.astype(np.int32))
        np.save(os.path.join(ldir, "token_slot_to_row.npy"), ts2row.reshape(T, K))

        toks = (ids_dst // K).astype(int)
        slots = (ids_dst % K).astype(int)
        etm = {}
        for e in range(E):
            s, t = int(bounds[e]), int(bounds[e + 1])
            etm[str(e)] = [{"token_id": int(toks[i]), "topk_slot": int(slots[i])}
                           for i in range(s, t)]
        json.dump(etm, open(os.path.join(ldir, "expert_token_map.json"), "w"),
                  ensure_ascii=False, indent=2)

        total = int(n_rows)
        experts = {str(e): {"count": int(counts[e]),
                            "ratio": float(counts[e]) / total if total else 0.0}
                   for e in range(E)}
        json.dump({"n_expert": E, "n_tokens": int(T), "topk": int(K),
                   "total_assignments": total, "experts": experts},
                  open(os.path.join(ldir, "expert_load_ratio.json"), "w"),
                  ensure_ascii=False, indent=2)

        print(f"L{L:02d} ops={len(ops_found)} T={T} rows={n_rows} "
              f"perm={is_perm} inverse={ok_inverse} expert={ok_expert} "
              f"counts={ok_counts} order={ok_order}{div_note}")

    print("=" * 64)
    if div_tokens_total:
        print(f"NOTE: fresh routing differs from previous router on "
              f"{div_tokens_count}/{div_tokens_total} (token,layer) rows "
              f"(expected: online run, code may have changed; not an error).")
    if fails == 0:
        print(f"VALIDATION PASSED — {len(layers)} layers, real capture is "
              f"internally consistent (router <-> reorder).")
        return 0
    print(f"VALIDATION FAILED — {fails} layer(s) inconsistent")
    return 1


if __name__ == "__main__":
    sys.exit(main())
