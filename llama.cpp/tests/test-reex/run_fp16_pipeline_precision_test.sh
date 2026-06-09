#!/usr/bin/env bash
# 完整精度测试：FP16 管线（含 K/V F16）+ backend-ops + PPL
# 用法:
#   ./tests/test-reex/run_fp16_pipeline_precision_test.sh [build_dir] [model_gguf] [data_file]
# 示例:
#   ./tests/test-reex/run_fp16_pipeline_precision_test.sh
#   ./tests/test-reex/run_fp16_pipeline_precision_test.sh build_ppl models/Qwen3-30B-A3B-Q4_K.gguf datasets/wikitext-103-raw-v1/validation.txt

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${1:-$REPO_ROOT/build}"
if [[ "$BUILD_DIR" != /* ]]; then
    BUILD_DIR="$REPO_ROOT/$BUILD_DIR"
fi
MODEL="${2:-}"
DATA="${3:-}"
REEX_GATE_PROFILE="${REEX_GATE_PROFILE:-full}"
VERIFY_CMAKE_GENERATOR="${VERIFY_CMAKE_GENERATOR:-}"
case "$REEX_GATE_PROFILE" in
    project|full) ;;
    *)
        echo "错误: 未知 REEX_GATE_PROFILE=$REEX_GATE_PROFILE（仅支持 project/full）" >&2
        exit 2
        ;;
esac
N_CHUNKS="${N_CHUNKS:-$([[ "$REEX_GATE_PROFILE" == "project" ]] && echo 32 || echo 64)}"
VERIFY_FP16_KVS="${VERIFY_FP16_KVS:-f16 q8q8}"
N_CTX=512
BATCH=512
NGL=99
ENABLE_CUDA=0
if [[ "${GGML_CUDA:-}" == "1" || "${GGML_CUDA:-}" == "ON" ]]; then
    ENABLE_CUDA=1
elif [[ "${SKIP_CUDA:-0}" != "1" ]] && command -v nvcc >/dev/null 2>&1; then
    ENABLE_CUDA=1
fi
RETRY_ATTEMPTS="${RETRY_ATTEMPTS:-2}"
PROJECT_CPU_SMOKE_OPS="${PROJECT_CPU_SMOKE_OPS:-MUL_MAT,SET_ROWS,SOFT_MAX,GELU_QUICK}"

format_duration() {
    local total_sec="${1:-0}"
    local h=$(( total_sec / 3600 ))
    local m=$(( (total_sec % 3600) / 60 ))
    local s=$(( total_sec % 60 ))
    if (( h > 0 )); then
        printf '%dh%02dm%02ds' "$h" "$m" "$s"
    elif (( m > 0 )); then
        printf '%dm%02ds' "$m" "$s"
    else
        printf '%ds' "$s"
    fi
}

require_cuda_runtime() {
    if [[ "$ENABLE_CUDA" != "1" ]]; then
        return 0
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        if nvidia-smi -L >/dev/null 2>&1; then
            return 0
        fi
        echo "错误: 已启用 CUDA 验证，但当前环境无法通过 nvidia-smi 访问 GPU。" >&2
        echo "  这通常意味着容器/会话失去了 GPU 句柄，继续执行会静默退回 CPU 路径。" >&2
        echo "  请先恢复 GPU 可见性后重试。" >&2
        exit 1
    fi

    if [[ -e /dev/nvidiactl && -e /dev/nvidia0 ]]; then
        return 0
    fi

    echo "错误: 已启用 CUDA 验证，但未检测到可用的 NVIDIA 运行时设备。" >&2
    exit 1
}

run_with_retry() {
    local label="$1"
    shift
    local attempt=1
    local code=0
    while (( attempt <= RETRY_ATTEMPTS )); do
        if (( RETRY_ATTEMPTS > 1 )); then
            echo "  $label（第 $attempt/$RETRY_ATTEMPTS 次）..."
        else
            echo "  $label..."
        fi
        if "$@"; then
            return 0
        fi
        code=$?
        echo "  $label 失败，exit_code=$code"
        if (( attempt == RETRY_ATTEMPTS )); then
            return "$code"
        fi
        echo "  5 秒后重试..."
        sleep 5
        attempt=$((attempt + 1))
    done
}

run_pipe_with_retry() {
    local label="$1"
    local out_file="$2"
    shift 2
    local attempt=1
    local code=0
    while (( attempt <= RETRY_ATTEMPTS )); do
        if (( RETRY_ATTEMPTS > 1 )); then
            echo "  $label（第 $attempt/$RETRY_ATTEMPTS 次）..."
        else
            echo "  $label..."
        fi
        set +e
        "$@" 2>&1 | tee "$out_file"
        code=${PIPESTATUS[0]}
        set -e
        if (( code == 0 )); then
            return 0
        fi
        echo "  $label 失败，exit_code=$code"
        if (( attempt == RETRY_ATTEMPTS )); then
            return "$code"
        fi
        echo "  5 秒后重试..."
        sleep 5
        attempt=$((attempt + 1))
    done
}

echo "=== FP16 Pipeline 完整精度测试 ==="
echo "  REEX_GATE_PROFILE: $REEX_GATE_PROFILE"
echo "  仓库根目录: $REPO_ROOT"
echo "  Build 目录: $BUILD_DIR"
echo "  模型: ${MODEL:-未指定，仅跑 backend-ops}"
echo "  数据: ${DATA:-未指定}"
echo "  chunks: $N_CHUNKS, n_ctx=$N_CTX, batch=$BATCH, ngl=$NGL"
echo "  VERIFY_FP16_KVS: $VERIFY_FP16_KVS"
echo "  VERIFY_CMAKE_GENERATOR: ${VERIFY_CMAKE_GENERATOR:-<cmake-default>}"
echo "  GGML_CUDA: $ENABLE_CUDA"
if [[ "$REEX_GATE_PROFILE" == "project" ]]; then
    echo "  PROJECT_CPU_SMOKE_OPS: $PROJECT_CPU_SMOKE_OPS"
fi
echo ""

require_cuda_runtime

# 1. 配置与编译（需 GGML_REEX_FP16_PIPELINE=ON）
echo "[1/4] 配置/刷新 CMake (GGML_REEX_FP16_PIPELINE=ON, GGML_USE_REEX=ON)..."
mkdir -p "$BUILD_DIR"
cmake_args=(
    -DGGML_REEX_FP16_PIPELINE=ON
    -DGGML_USE_REEX=ON
    -DCMAKE_BUILD_TYPE=Release
)
if [[ -n "$VERIFY_CMAKE_GENERATOR" ]]; then
    cmake_args=( -G "$VERIFY_CMAKE_GENERATOR" "${cmake_args[@]}" )
fi
if [[ "$ENABLE_CUDA" == "1" ]]; then
    cmake_args+=( -DGGML_CUDA=ON )
    enable_fa_all_quants=0
    if [[ "${VERIFY_FP16_FORCE_FA_ALL_QUANTS:-0}" == "1" ]]; then
        enable_fa_all_quants=1
    elif [[ -n "$MODEL" && -n "$DATA" ]]; then
        case " $VERIFY_FP16_KVS " in
            *" k8v4 "*|*" q8q8 "*) enable_fa_all_quants=1 ;;
        esac
    fi
    if (( enable_fa_all_quants == 1 )); then
        cmake_args+=( -DGGML_CUDA_FA_ALL_QUANTS=ON )
        echo "  FP16 CMake: 启用 GGML_CUDA_FA_ALL_QUANTS=ON"
    else
        echo "  FP16 CMake: 跳过 GGML_CUDA_FA_ALL_QUANTS（当前用例不需要）"
    fi
    if [[ -n "${VERIFY_CUDA_ARCHITECTURES:-}" ]]; then
        cmake_args+=( "-DCMAKE_CUDA_ARCHITECTURES=${VERIFY_CUDA_ARCHITECTURES}" )
    fi
fi
cmake -S "$REPO_ROOT" -B "$BUILD_DIR" "${cmake_args[@]}"
CACHE="$BUILD_DIR/CMakeCache.txt"
if ! grep -q "GGML_REEX_FP16_PIPELINE:BOOL=ON" "$CACHE" 2>/dev/null; then
    echo "错误: CMake 配置后未启用 GGML_REEX_FP16_PIPELINE。"
    exit 1
fi

echo "[2/4] 编译 test-backend-ops 与 llama-perplexity..."
cmake --build "$BUILD_DIR" -j"${VERIFY_JOBS:-${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc 2>/dev/null || echo 4)}}" --target test-backend-ops llama-perplexity

BIN_OPS="$BUILD_DIR/test-backend-ops"
for p in "$BUILD_DIR/test-backend-ops" "$BUILD_DIR/tests/test-backend-ops" "$BUILD_DIR/bin/test-backend-ops"; do
    [ -x "$p" ] && BIN_OPS="$p" && break
done
if [ ! -x "$BIN_OPS" ]; then
    echo "错误: 未找到可执行的 test-backend-ops，请检查 build 目录。"
    exit 1
fi

BIN_PPL="$BUILD_DIR/bin/llama-perplexity"
[ ! -x "$BIN_PPL" ] && BIN_PPL="$BUILD_DIR/llama-perplexity"

# 2. Backend-ops 测试（CPU + CUDA 双端验证）
# 默认模式：若有 CUDA，则跑 非CPU backend（如 CUDA）与 CPU(ref) 一致性；若仅 CPU 构建则默认会跳过 CPU，需显式跑 CPU。
echo ""
echo "[3/4] 运行 test-backend-ops（含 rms_norm F16、softmax F16、mul_mat F16 等；ggml_set_rows 仍为 F32 源，与上游一致）..."
run_with_retry "test-backend-ops" "$BIN_OPS"
echo "  test-backend-ops 默认通过."
# CPU-only 构建时默认会跳过 CPU backend，显式跑 CPU 以验证 FP16 管线在 CPU 上的正确性。
# project 档只保留一小组核心算子，避免门禁在整套 CPU backend-ops 上超时。
if ! grep -Eq "^GGML_CUDA:.*=ON$" "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
    if [[ "$REEX_GATE_PROFILE" == "project" ]]; then
        echo "  (CPU-only 构建) 额外运行精简 CPU smoke: $PROJECT_CPU_SMOKE_OPS"
        run_with_retry "test-backend-ops -b CPU -o $PROJECT_CPU_SMOKE_OPS" \
            "$BIN_OPS" test -b CPU -o "$PROJECT_CPU_SMOKE_OPS"
    else
        echo "  (CPU-only 构建) 额外运行 test-backend-ops -b CPU 做 CPU 端验证..."
        run_with_retry "test-backend-ops -b CPU" "$BIN_OPS" -b CPU
    fi
fi
echo "  test-backend-ops 通过."
echo ""

# 3. PPL 测试（若提供模型与数据）
RESULTS_DIR="$SCRIPT_DIR/ppl_fp16_pipeline_results"
if [ -n "$MODEL" ] && [ -n "$DATA" ]; then
    if [ ! -x "$BIN_PPL" ]; then
        echo "错误: 未找到 llama-perplexity，请检查 build 目录。"
        exit 1
    fi
    echo "[4/4] PPL 精度测试（FP16 管线 + K/V F16）..."
    mkdir -p "$RESULTS_DIR"
    for kv in $VERIFY_FP16_KVS; do
        FA_ARGS=()
        case "$kv" in
            f16)  CTK=();;
            k8v4) CTK=(-ctk q8_0 -ctv q4_0); FA_ARGS=(--flash-attn on) ;;
            q8q8) CTK=(-ctk q8_0 -ctv q8_0); FA_ARGS=(--flash-attn on) ;;
            *)
                echo "错误: 未知 VERIFY_FP16_KVS 项: $kv（支持 f16/k8v4/q8q8）"
                exit 2
                ;;
        esac
        OUT="$RESULTS_DIR/ppl_fp16_kv_${kv}_n${N_CHUNKS}.txt"
        echo "  KV=$kv -> $OUT"
        kv_start_epoch="$(date +%s)"
        run_pipe_with_retry "llama-perplexity KV=$kv" "$OUT" \
            "$BIN_PPL" -m "$MODEL" -f "$DATA" --chunks "$N_CHUNKS" --ctx-size "$N_CTX" --batch-size "$BATCH" \
            -ngl "$NGL" "${CTK[@]}" "${FA_ARGS[@]}" -t 1
        echo "  KV=$kv 完成，耗时: $(format_duration $(( $(date +%s) - kv_start_epoch )))"
    done
    echo ""
    echo "PPL 结果已写入 $RESULTS_DIR/"
    grep -h "perplexity:" "$RESULTS_DIR"/ppl_fp16_kv_*.txt 2>/dev/null || true
else
    echo "[4/4] 未提供模型/数据，跳过 PPL。若要跑 PPL 请执行："
    echo "  MODEL=/path/to/Qwen3-30B-A3B-Q4_K.gguf DATA=/path/to/wiki.raw $0 $BUILD_DIR \$MODEL \$DATA"
fi

echo ""
echo "=== 完整精度测试完成 ==="
