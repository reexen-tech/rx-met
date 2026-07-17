#!/usr/bin/env bash
# 把 output/legacy 下所有 case 转成 hex CSV。
#   env: SRC=<case根目录>  OUT=<输出根目录>  CHECK_ROWS=<自校验行数>  OUTPUTS=<ALL|none|F16,...>
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAGEN_ROOT="$(cd "$HERE/../.." && pwd)"                 # tools/reex-gemm-datagen
SRC="${SRC:-$DATAGEN_ROOT/output/legacy}"
OUT="${OUT:-$HERE/output}"
CHECK_ROWS="${CHECK_ROWS:-64}"
OUTPUTS="${OUTPUTS:-ALL}"
export PYTHONUNBUFFERED=1

mapfile -t CASES < <(find "$SRC" -mindepth 1 -maxdepth 1 -type d | sort)
echo "[run] ${#CASES[@]} cases from $SRC  -> $OUT"

for c in "${CASES[@]}"; do
    python3 "$HERE/reex_bin_to_csv.py" "$c" --out "$OUT" --outputs "$OUTPUTS" --check-rows "$CHECK_ROWS"
done

echo "[all] done -> $OUT"
du -sh "$OUT"/* 2>/dev/null | tail -n +1 || true
