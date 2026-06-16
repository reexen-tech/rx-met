#!/usr/bin/env bash
# Wait until the shared GPU has enough free memory, then launch the full
# 6-case MoE prefill capture pipeline. Model needs ~18 GB + compute buffers.
set -u
NEED_MIB=${NEED_MIB:-20000}
LOG=/tmp/moe_full2.log
cd /mnt/data8t/zcx/aimet_rx/llama.cpp

echo "=== WAIT start $(date), need ${NEED_MIB} MiB free ===" >> "$LOG"
while true; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [ -n "$FREE" ] && [ "$FREE" -ge "$NEED_MIB" ]; then
        echo "=== GPU free=${FREE} MiB >= ${NEED_MIB}, launching $(date) ===" >> "$LOG"
        break
    fi
    sleep 60
done

bash tests/test-reex/run_full.sh >> "$LOG" 2>&1
