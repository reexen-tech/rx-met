#!/usr/bin/env bash
# Qwen3Moe 全层对比：构建 5 种配置，在不同 KV cache 类型下对比
# KV cache 类型：f16（默认）、q8_0、q4_0
# 输出：Layer 0 中间 tensor 对比 + final_logits（模型最终输出）对比
#
# ─── 各配置说明 ───
# 1) gold（build_gold）
#    模型：F16。REEX_GEMM=OFF，USE_REEX=OFF。
#    矩阵乘：原生 float×float。sin/cos：系统 libm。
#    用途：金标基线，与其余 4 组对比。
#
# 2) W4×float（build_w4f32）
#    模型：Q4_K。REEX_GEMM=ON，ACTIVATION=FLOAT，USE_REEX=OFF。
#    矩阵乘：REEX GEMM，权重 Q4_K × 激活 float。sin/cos：libm。
#    用途：REX 量化 GEMM 的精度参考路径。
#
# 3) W4×INT16（build_w4q16）
#    模型：Q4_K。REEX_GEMM=ON，ACTIVATION=Q16，USE_REEX=OFF。
#    矩阵乘：REEX GEMM，权重 Q4_K × 激活 INT16（每 block scale）。sin/cos：libm。
#    用途：REX 中等精度/带宽路径。
#
# 4) W4×INT8（build_w4q8）
#    模型：Q4_K。REEX_GEMM=ON，ACTIVATION=Q8，USE_REEX=OFF。
#    矩阵乘：REEX GEMM，权重 Q4_K × 激活 INT8（每 block scale）。sin/cos：libm。
#    用途：REX 低精度/带宽路径。
#
# 5) gold_reex_lut（build_gold_reex_lut，新增）
#    模型：F16。REEX_GEMM=OFF，USE_REEX=ON。
#    矩阵乘：同上 gold，原生 float×float。sin/cos：REEX 分段线性 LUT。
#    用途：仅替换 sin/cos 为 LUT，与 gold 对比可隔离 LUT 的端到端影响。
#
# ─── 本脚本用到的 CMake 宏 ───
# GGML_REEX_GEMM     OFF=原生矩阵乘，ON=REX GEMM（激活格式由下一项定）。
# GGML_REEX_GEMM_ACTIVATION   FLOAT | Q16 | Q8（仅当 REEX_GEMM=ON 时有效）。
# GGML_USE_REEX      OFF=sin/cos 用 libm，ON=sin/cos 用 REX 分段线性 LUT。
# GGML_CPU_REPACK    OFF（REX 路径下关闭 CPU repack）。
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_CPP_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_F16="${MODEL_F16:-$LLAMA_CPP_ROOT/models/shared_models/Qwen/Qwen3-30B-A3B-Instruct-2507/Qwen3-30B-A3B-Instruct-2507-F16.gguf}"
MODEL_Q4_K="${MODEL_Q4_K:-$LLAMA_CPP_ROOT/models/shared_models/Qwen/Qwen3-30B-A3B-Instruct-2507/Qwen3-30B-A3B-Instruct-2507-Q4_K.gguf}"
BUILD_BASE="${BUILD_BASE:-$LLAMA_CPP_ROOT/build_single_block}"
DUMP_BASE="${DUMP_BASE:-$SCRIPT_DIR/dump_single_block}"
PROMPT="${PROMPT:-The quick brown fox jumps over the lazy dog}"
N_THREADS="${N_THREADS:-4}"
KV_TYPES="${KV_TYPES:-f16 q8_0 q4_0}"

echo "=================================================================="
echo "  Qwen3Moe full-layer comparison (REEX all layers)"
echo "=================================================================="
echo "  MODEL_F16=$MODEL_F16"
echo "  MODEL_Q4_K=$MODEL_Q4_K"
echo "  BUILD_BASE=$BUILD_BASE"
echo "  DUMP_BASE=$DUMP_BASE"
echo "  PROMPT=$PROMPT"
echo "  KV_TYPES=$KV_TYPES"
echo ""

if [[ ! -f "$MODEL_F16" ]]; then echo "ERROR: F16 model not found: $MODEL_F16"; exit 1; fi
if [[ ! -f "$MODEL_Q4_K" ]]; then echo "ERROR: Q4_K model not found: $MODEL_Q4_K"; exit 1; fi

mkdir -p "$BUILD_BASE" "$DUMP_BASE"

# touch ggml-cpu.c 以强制 cmake 增量构建在编译定义变化时重编 ggml-cpu
touch "$LLAMA_CPP_ROOT/ggml/src/ggml-cpu/ggml-cpu.c"

########################################################################
# Build 5 configurations (only once, shared across all KV types)
########################################################################

# 1) gold：F16 模型，原生 float×float + libm sin/cos，金标基线
BUILD_GOLD="$BUILD_BASE/build_gold"
echo "--- Build 1/5: gold (F16, REEX_GEMM=OFF, USE_REEX=OFF) ---"
cmake -B "$BUILD_GOLD" -S "$LLAMA_CPP_ROOT" \
    -DGGML_REEX_GEMM=OFF \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_GOLD" --target eval_single_token -j

# 2) W4×float：Q4_K 模型，REX GEMM 激活=FLOAT，sin/cos=libm
BUILD_W4F32="$BUILD_BASE/build_w4f32"
echo "--- Build 2/5: W4×float (Q4_K, REEX_GEMM=ON FLOAT, USE_REEX=OFF) ---"
cmake -B "$BUILD_W4F32" -S "$LLAMA_CPP_ROOT" \
    -DGGML_REEX_GEMM=ON \
    -DGGML_REEX_GEMM_ACTIVATION=FLOAT \
    -DGGML_CPU_REPACK=OFF \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_W4F32" --target eval_single_token -j

# 3) W4×INT16：Q4_K 模型，REX GEMM 激活=Q16，sin/cos=libm
BUILD_W4Q16="$BUILD_BASE/build_w4q16"
echo "--- Build 3/5: W4×INT16 (Q4_K, REEX_GEMM=ON Q16, USE_REEX=OFF) ---"
cmake -B "$BUILD_W4Q16" -S "$LLAMA_CPP_ROOT" \
    -DGGML_REEX_GEMM=ON \
    -DGGML_REEX_GEMM_ACTIVATION=Q16 \
    -DGGML_CPU_REPACK=OFF \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_W4Q16" --target eval_single_token -j

# 4) W4×INT8：Q4_K 模型，REX GEMM 激活=Q8，sin/cos=libm
BUILD_W4Q8="$BUILD_BASE/build_w4q8"
echo "--- Build 4/5: W4×INT8 (Q4_K, REEX_GEMM=ON Q8, USE_REEX=OFF) ---"
cmake -B "$BUILD_W4Q8" -S "$LLAMA_CPP_ROOT" \
    -DGGML_REEX_GEMM=ON \
    -DGGML_REEX_GEMM_ACTIVATION=Q8 \
    -DGGML_CPU_REPACK=OFF \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_W4Q8" --target eval_single_token -j

# 5) gold_reex_lut：F16 模型，仅 sin/cos 改为 REX LUT（新增配置）
BUILD_GOLD_LUT="$BUILD_BASE/build_gold_reex_lut"
echo "--- Build 5/5: gold_reex_lut (F16, REEX_GEMM=OFF, USE_REEX=ON) ---"
cmake -B "$BUILD_GOLD_LUT" -S "$LLAMA_CPP_ROOT" \
    -DGGML_REEX_GEMM=OFF \
    -DGGML_USE_REEX=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_GOLD_LUT" --target eval_single_token -j

# Resolve executable paths
EVAL_GOLD="$BUILD_GOLD/bin/eval_single_token"
[[ -x "$EVAL_GOLD" ]] || EVAL_GOLD="$BUILD_GOLD/tests/eval_single_token"
EVAL_GOLD_LUT="$BUILD_GOLD_LUT/bin/eval_single_token"
[[ -x "$EVAL_GOLD_LUT" ]] || EVAL_GOLD_LUT="$BUILD_GOLD_LUT/tests/eval_single_token"
EVAL_W4F32="$BUILD_W4F32/bin/eval_single_token"
[[ -x "$EVAL_W4F32" ]] || EVAL_W4F32="$BUILD_W4F32/tests/eval_single_token"
EVAL_W4Q16="$BUILD_W4Q16/bin/eval_single_token"
[[ -x "$EVAL_W4Q16" ]] || EVAL_W4Q16="$BUILD_W4Q16/tests/eval_single_token"
EVAL_W4Q8="$BUILD_W4Q8/bin/eval_single_token"
[[ -x "$EVAL_W4Q8" ]] || EVAL_W4Q8="$BUILD_W4Q8/tests/eval_single_token"

for exe in "$EVAL_GOLD" "$EVAL_W4F32" "$EVAL_W4Q16" "$EVAL_W4Q8" "$EVAL_GOLD_LUT"; do
    if [[ ! -x "$exe" ]]; then echo "ERROR: not found: $exe"; exit 1; fi
done

########################################################################
# Helper: run one eval and show key output lines
########################################################################
run_eval() {
    local label="$1" logfile="$2" ; shift 2
    echo "--- Run $label ---"
    "$@" 2>"$logfile" && true
    local rc=$?
    grep -E "Decode|ERROR|Saved final|Top-5|token=|KV cache|REEX CUDA trace|q8_mul_mat_hits|q8_mul_mat_id_hits|q16_mul_mat_hits|q16_batch_fallbacks|q16_mul_mat_id_hits|q16_mul_mat_id_unsupported" "$logfile" || true
    if [ $rc -ne 0 ]; then echo "ERROR: $label exited with code $rc"; exit $rc; fi
    echo ""
}

########################################################################
# Run + Compare for each KV cache type
########################################################################
OVERALL_RC=0

for KV in $KV_TYPES; do
    echo ""
    echo "=================================================================="
    echo "  KV cache type: $KV"
    echo "=================================================================="

    KV_SUFFIX=""
    KV_ARGS=""
    if [[ "$KV" != "f16" ]]; then
        KV_SUFFIX="_kv${KV}"
        KV_ARGS="-ctk $KV -ctv $KV"
    fi

    DUMP_DIR="$DUMP_BASE/kv_${KV}"
    LOG_DIR="$DUMP_DIR/logs"
    mkdir -p "$DUMP_DIR/dump_f32f32" "$DUMP_DIR/dump_f32f32_lut" "$DUMP_DIR/dump_w4f32" \
             "$DUMP_DIR/dump_w4q16" "$DUMP_DIR/dump_w4q8" "$LOG_DIR"

    run_eval "gold (F16, KV=$KV)" "$LOG_DIR/gold.log" \
        env REEX_DUMP_DIR="$DUMP_DIR/dump_f32f32" REEX_DUMP_LAYER=0 \
        "$EVAL_GOLD" -m "$MODEL_F16" -t "$N_THREADS" -p "$PROMPT" $KV_ARGS

    run_eval "W4×float (all layers REEX FLOAT, KV=$KV)" "$LOG_DIR/w4f32.log" \
        env REEX_DUMP_DIR="$DUMP_DIR/dump_w4f32" REEX_DUMP_LAYER=0 \
        "$EVAL_W4F32" -m "$MODEL_Q4_K" -t "$N_THREADS" -p "$PROMPT" $KV_ARGS

    run_eval "W4×INT16 (all layers REEX Q16, KV=$KV)" "$LOG_DIR/w4q16.log" \
        env REEX_DUMP_DIR="$DUMP_DIR/dump_w4q16" REEX_DUMP_LAYER=0 \
        "$EVAL_W4Q16" -m "$MODEL_Q4_K" -t "$N_THREADS" -p "$PROMPT" $KV_ARGS

    run_eval "W4×INT8 (all layers REEX Q8, KV=$KV)" "$LOG_DIR/w4q8.log" \
        env REEX_DUMP_DIR="$DUMP_DIR/dump_w4q8" REEX_DUMP_LAYER=0 \
        "$EVAL_W4Q8" -m "$MODEL_Q4_K" -t "$N_THREADS" -p "$PROMPT" $KV_ARGS

    run_eval "gold+REEX_LUT (F16, sin/cos LUT, KV=$KV)" "$LOG_DIR/gold_lut.log" \
        env REEX_DUMP_DIR="$DUMP_DIR/dump_f32f32_lut" REEX_DUMP_LAYER=0 \
        "$EVAL_GOLD_LUT" -m "$MODEL_F16" -t "$N_THREADS" -p "$PROMPT" $KV_ARGS

    echo "--- Compare KV=$KV (gold vs W4×float, vs W4×INT16, vs W4×INT8, vs gold+REEX_LUT) ---"
    cd "$SCRIPT_DIR"
    python3 compare_layer_dumps.py \
        --gold "$DUMP_DIR/dump_f32f32" \
        --test "$DUMP_DIR/dump_w4f32" "W4×float" \
        --test "$DUMP_DIR/dump_w4q16" "W4×INT16" \
        --test "$DUMP_DIR/dump_w4q8" "W4×INT8" \
        --test "$DUMP_DIR/dump_f32f32_lut" "gold+REEX_LUT" \
        && true
    local_rc=$?
    if [ $local_rc -ne 0 ]; then OVERALL_RC=$local_rc; fi
done

echo ""
echo "=================================================================="
echo "  All KV cache types tested: $KV_TYPES"
echo "=================================================================="
exit $OVERALL_RC
