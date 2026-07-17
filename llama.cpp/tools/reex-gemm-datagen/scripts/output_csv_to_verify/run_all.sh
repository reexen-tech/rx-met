#!/usr/bin/env bash
# 把 reex-gemm-datagen 的三组 case 全部转成 hex CSV, 分到三个子文件夹:
#   output/legacy 下的 case  -> output_csv_to_verify/output/legacy-quant/<case>/
#   output/kquant 下的 case  -> output_csv_to_verify/output/k-quant/<case>/
#   output/int    下的 case  -> output_csv_to_verify/output/int/<case>/
# env: OUTPUTS=<ALL|none|F16,...>  CHECK_ROWS=<n>
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAGEN_ROOT="$(cd "$HERE/../.." && pwd)"                 # tools/reex-gemm-datagen
OUT_ROOT="$HERE/output"
OUTPUTS="${OUTPUTS:-ALL}"
CHECK_ROWS="${CHECK_ROWS:-64}"
export PYTHONUNBUFFERED=1

for src_name in legacy kquant int; do
    case "$src_name" in
        legacy) group="legacy-quant" ;;
        kquant) group="k-quant" ;;
        int)    group="int" ;;
    esac
    src="$DATAGEN_ROOT/output/$src_name"
    dst="$OUT_ROOT/$group"
    [ -d "$src" ] || { echo "[skip] $src 不存在"; continue; }
    mapfile -t CASES < <(find "$src" -mindepth 1 -maxdepth 1 -type d | sort)
    echo "==== group $src_name -> $dst  (${#CASES[@]} cases) ===="
    mkdir -p "$dst"
    for c in "${CASES[@]}"; do
        cname="$(basename "$c")"
        # 断点续跑: meta.json 是每个 case 最后写的, 存在即已完成 -> 跳过
        if [ -f "$dst/$cname/meta.json" ]; then
            echo "[skip] $cname (已完成)"
            continue
        fi
        python3 "$HERE/reex_bin_to_csv.py" "$c" --out "$dst" --outputs "$OUTPUTS" --check-rows "$CHECK_ROWS"
    done
done

echo "[all] done -> $OUT_ROOT"
du -sh "$OUT_ROOT"/* 2>/dev/null || true
