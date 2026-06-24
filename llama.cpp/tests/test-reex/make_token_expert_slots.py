#!/usr/bin/env python3
"""make_token_expert_slots.py — per-layer token-centric view of the expert
reorder.

For every output/case_XX/layer_LL/ it writes token_expert_slots.json:

  {
    "<token_id>": {
      "<slot>": {"expert_id": <e>, "slot_in_expert": <local_row>},
      ...   # slot = 0 .. K-1 (该专家在此 token 的 top-K 名次)
    },
    ...
  }

slot_in_expert = token_slot_to_row[token, slot] - expert_bounds[expert_id],
i.e. the row this (token, expert) result occupies INSIDE that expert's own
computed output batch (0-based). Derived purely from the captured artifacts
(router_expert_ids / token_slot_to_row / expert_bounds).
"""
import json
import os
import sys

import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else \
    "/mnt/data8t/zcx/aimet_rx/llama.cpp/output/case_00"


def main():
    layer_dirs = sorted(d for d in os.listdir(ROOT)
                        if d.startswith("layer_") and
                        os.path.isdir(os.path.join(ROOT, d)))
    if not layer_dirs:
        print(f"ERROR: no layer_* dirs under {ROOT}")
        return 1

    for d in layer_dirs:
        ldir = os.path.join(ROOT, d)
        ids = np.load(os.path.join(ldir, "router_expert_ids.npy"))      # [T, K]
        ts2row = np.load(os.path.join(ldir, "token_slot_to_row.npy"))   # [T, K]
        bounds = np.load(os.path.join(ldir, "expert_bounds.npy"))       # [E+1]
        T, K = ids.shape

        # local row inside each expert's own batch (vectorized)
        local = (ts2row - bounds[ids]).astype(np.int64)

        out = {}
        ids_l = ids.tolist()
        loc_l = local.tolist()
        for t in range(T):
            row = ids_l[t]
            lrow = loc_l[t]
            out[str(t)] = {
                str(slot): {"expert_id": int(row[slot]),
                            "slot_in_expert": int(lrow[slot])}
                for slot in range(K)
            }

        with open(os.path.join(ldir, "token_expert_slots.json"), "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"{d}: T={T} K={K} -> token_expert_slots.json")

    print(f"done: {len(layer_dirs)} layers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
