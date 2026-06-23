#!/usr/bin/env bash
# Qwen3.5-35B-A3B block-64 逐层 Tensor 精度验证入口（Docker GPU）
#
# 宿主机调用:
#   ./tests/test-reex/run_qwen35_block64_gpu_docker.sh build
#   ./tests/test-reex/run_qwen35_block64_gpu_docker.sh dump
#   ./tests/test-reex/run_qwen35_block64_gpu_docker.sh compare
#
# 本脚本覆盖：Tensor 逐层 dump / compare / 硬件采数 / block-64 PPL（wikitext validation）。
# 不跑 MoE 专项，不做 CPU 对比。
set -euo pipefail

CONTAINER="${BLOCK64_CONTAINER:-quant-gru-cuda128}"
WORKDIR="${BLOCK64_WORKDIR:-/mnt/data8t/zcx/aimet_rx/llama.cpp}"
BUILD_DIR="${BLOCK64_BUILD_DIR:-build_cuda_q64}"
OUT_ROOT="${BLOCK64_OUT:-$WORKDIR/tests/test-reex/results/qwen35b_a3b_block64_tensor}"
MODEL_F16="${MODEL_F16:-/mnt/data8t/share/models/Qwen/Qwen3.5-35B-A3B/Qwen3.5-35B-A3B-f16.gguf}"
TYPE="${BLOCK64_TYPE:-Q4_0_64}"
MODEL_Q64="${MODEL_Q64:-/mnt/data8t/share/models/Qwen/Qwen3.5-35B-A3B/Qwen3.5-35B-A3B-${TYPE}.gguf}"
PROMPT_FILE="${BLOCK64_PROMPT_FILE:-$OUT_ROOT/prompt_seed.txt}"
CTX="${BLOCK64_CTX:-8192}"
BATCH="${BLOCK64_BATCH:-1024}"
NTOK="${BLOCK64_NTOK:-8192}"
DUMP_LAYER="${BLOCK64_DUMP_LAYER:--1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NGPU_LAYERS="${BLOCK64_NGL:-99}"
NGL_GOLD="${BLOCK64_NGL_GOLD:-8}"   # 35B F16 ~65GB；单卡 32GB 须部分 offload
NGL_TEST="${BLOCK64_NGL_TEST:-99}"  # Q*_64 ~18GB，可全 GPU
JOBS="${BLOCK64_JOBS:-16}"
B="${REEX_Q64_PSUM_BITS:-off}"

usage() {
    cat <<EOF
Usage: $0 <build|build-quantize|quantize|dump|dump-test-only|dump-smoke|compare|ppl|hw-dump-smoke|hw-dump|hw-dump-worst|hw-dump-phase2|hw-validate|hw-export|shell>

  build          配置并编译 eval_single_token（GPU + REEX_Q64 + FP16 pipeline）
  build-quantize 编译 llama-quantize（weight sweep 量化用）
  quantize       从 F16 量化单个 Q*_64 GGUF（需 BLOCK64_TYPE=...）
  dump           dump F16 Gold 与 Q*_64 Test 的逐层 Tensor（Prefill，默认 8192 tok）
  dump-test-only 仅 dump Test（复用已有 F16 Gold，weight sweep 用）
  dump-smoke     短 prompt 全层 dump（验证 pipeline，1 token）
  compare        使用 compare_layer_dumps_v2.py 对比 dump 结果
  ppl            wikitext-103 validation PPL（ctx=512, batch=512, chunks=-1）
  hw-dump-smoke  硬件采数 Phase 1 smoke（REEX_HW_DUMP_* env + meta.json）
  hw-dump        硬件采数 8k prefill（单 layer，默认 L19 B=8）
  hw-dump-worst  硬件采数 8k prefill L19/L23/L39 三连跑（默认 ffn_moe_up）
  hw-dump-phase2 硬件采数 Phase 2：ffn_moe_down @ L19/L23/L39（8k prefill, B=8）
  hw-validate    批量校验 hw_vectors_* 下全部 case（output vs sw float）
  hw-export      导出 RTL bundle（manifest.json + cases/）到 hw_vectors_rtl_bundle/
  shell          进入容器工作目录 bash

环境变量:
  BLOCK64_CONTAINER=$CONTAINER
  BLOCK64_WORKDIR=$WORKDIR
  MODEL_F16=$MODEL_F16
  MODEL_Q64=$MODEL_Q64
  BLOCK64_TYPE=$TYPE
  BLOCK64_CTX=$CTX
  BLOCK64_BATCH=$BATCH
  BLOCK64_NTOK=$NTOK
  BLOCK64_DUMP_LAYER=$DUMP_LAYER
  BLOCK64_PROMPT_FILE=$PROMPT_FILE
  BLOCK64_JOBS=$JOBS
  BLOCK64_NGL_GOLD=$NGL_GOLD
  BLOCK64_NGL_TEST=$NGL_TEST
  REEX_Q64_PSUM_BITS=${REEX_Q64_PSUM_BITS:-<unset/off>}
  PPL_DATASET=${PPL_DATASET:-/mnt/data8t/share/datasets/.../validation.txt}
  PPL_CTX=${PPL_CTX:-512}  PPL_BATCH=${PPL_BATCH:-512}  PPL_CHUNKS=${PPL_CHUNKS:--1}
  REEX_HW_DUMP_DIR=${REEX_HW_DUMP_DIR:-<unset>}
  REEX_HW_DUMP_TENSOR=${REEX_HW_DUMP_TENSOR:-<unset>}
  REEX_HW_DUMP_LAYER=${REEX_HW_DUMP_LAYER:--1}
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi

CMD="$1"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "错误: 容器「$CONTAINER」未运行。请先 docker start $CONTAINER" >&2
    exit 1
fi

run_in_container() {
    docker exec \
        -e CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        -e REEX_Q64_PSUM_BITS="${REEX_Q64_PSUM_BITS:-}" \
        "$CONTAINER" bash -lc "$1"
}

case "$CMD" in
    build)
        run_in_container "
set -euo pipefail
cd '$WORKDIR'
export PATH=/usr/local/cuda/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}
cmake -S . -B '$BUILD_DIR' \\
  -DCMAKE_BUILD_TYPE=Release \\
  -DGGML_CUDA=ON \\
  -DGGML_USE_REEX_Q64=ON \\
  -DGGML_REEX_FP16_PIPELINE=ON \\
  -DGGML_USE_REEX=ON
cmake --build '$BUILD_DIR' -j'$JOBS' --target eval_single_token
test -x '$BUILD_DIR'/bin/eval_single_token
echo '=== eval_single_token build ok ==='
"
        ;;
    build-quantize)
        run_in_container "
set -euo pipefail
cd '$WORKDIR'
cmake -S . -B '$BUILD_DIR' \\
  -DCMAKE_BUILD_TYPE=Release \\
  -DGGML_CUDA=ON \\
  -DGGML_USE_REEX_Q64=ON \\
  -DGGML_REEX_FP16_PIPELINE=ON \\
  -DGGML_USE_REEX=ON
cmake --build '$BUILD_DIR' -j'$JOBS' --target llama-quantize
test -x '$BUILD_DIR'/bin/llama-quantize
echo '=== llama-quantize build ok ==='
"
        ;;
    quantize)
        run_in_container "
set -euo pipefail
cd '$WORKDIR'
test -x '$BUILD_DIR'/bin/llama-quantize || { echo '先跑 build-quantize' >&2; exit 2; }
test -s '$MODEL_F16' || { echo '缺少 F16: $MODEL_F16' >&2; exit 2; }
OUT_GGUF='$MODEL_Q64'
if [[ -s \"\$OUT_GGUF\" ]]; then
  echo \"已存在，跳过: \$OUT_GGUF\"
  exit 0
fi
echo \"=== quantize $TYPE: $MODEL_F16 -> \$OUT_GGUF ===\"
'$BUILD_DIR'/bin/llama-quantize '$MODEL_F16' \"\$OUT_GGUF\" '$TYPE' '$JOBS'
ls -lh \"\$OUT_GGUF\"
"
        ;;
    dump|dump-smoke|dump-test-only)
        if [[ "$CMD" == dump-smoke ]]; then
            RUN_CTX=512; RUN_BATCH=512; RUN_NTOK=1; RUN_LAYER=-1
            GOLD_SUB=dump_f16_smoke
            TEST_SUB="dump_${TYPE}_B${B}_smoke"
            SKIP_GOLD=0
        elif [[ "$CMD" == dump-test-only ]]; then
            RUN_CTX=$CTX; RUN_BATCH=$BATCH; RUN_NTOK=$NTOK; RUN_LAYER=$DUMP_LAYER
            GOLD_SUB="dump_f16_prefill${NTOK}"
            TEST_SUB="dump_${TYPE}_B${B}_prefill${NTOK}"
            SKIP_GOLD=1
        else
            RUN_CTX=$CTX; RUN_BATCH=$BATCH; RUN_NTOK=$NTOK; RUN_LAYER=$DUMP_LAYER
            GOLD_SUB="dump_f16_prefill${NTOK}"
            TEST_SUB="dump_${TYPE}_B${B}_prefill${NTOK}"
            SKIP_GOLD=0
        fi
        run_in_container "
set -euo pipefail
cd '$WORKDIR'
export PATH=/usr/local/cuda/bin:\$PATH
export LD_LIBRARY_PATH='$BUILD_DIR'/bin:/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}
mkdir -p '$OUT_ROOT'
if [[ ! -s '$PROMPT_FILE' ]]; then
  cat > '$PROMPT_FILE' <<'SEED'
The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.
How vexingly quick daft zebras jump! Sphinx of black quartz, judge my vow.
SEED
fi
GOLD_DIR='$OUT_ROOT/$GOLD_SUB'
TEST_DIR='$OUT_ROOT/$TEST_SUB'
EVAL_ARGS=\"-f $PROMPT_FILE -c $RUN_CTX -b $RUN_BATCH --n-tokens $RUN_NTOK\"
if [[ $SKIP_GOLD -eq 0 ]]; then
  rm -rf \"\$GOLD_DIR\" \"\$TEST_DIR\"
else
  rm -rf \"\$TEST_DIR\"
  if [[ ! -d \"\$GOLD_DIR\" ]]; then
    echo '错误: 缺少 F16 Gold '$OUT_ROOT/$GOLD_SUB'，请先跑 dump' >&2
    exit 2
  fi
fi
if [[ $SKIP_GOLD -eq 0 ]]; then
echo \"=== F16 Gold (ngl=$NGL_GOLD, layer=$RUN_LAYER, $RUN_NTOK tok) ===\"
REEX_DUMP_DIR=\"\$GOLD_DIR\" REEX_DUMP_LAYER=$RUN_LAYER \\
  '$BUILD_DIR'/bin/eval_single_token -m '$MODEL_F16' -ngl '$NGL_GOLD' \$EVAL_ARGS
else
echo \"=== F16 Gold: reuse \$GOLD_DIR ===\"
fi
echo \"=== Test (ngl=$NGL_TEST, layer=$RUN_LAYER, $RUN_NTOK tok) ===\"
REEX_DUMP_DIR=\"\$TEST_DIR\" REEX_DUMP_LAYER=$RUN_LAYER REEX_Q64_PSUM_BITS='${REEX_Q64_PSUM_BITS:-}' \\
  '$BUILD_DIR'/bin/eval_single_token -m '$MODEL_Q64' -ngl '$NGL_TEST' \$EVAL_ARGS
echo \"=== dump ok ===\"
echo \"GOLD_DIR=\$GOLD_DIR\"
echo \"TEST_DIR=\$TEST_DIR\"
"
        ;;
    compare)
        run_in_container "
set -euo pipefail
cd '$WORKDIR'
SCRIPT=tests/test-reex/compare_layer_dumps_v2.py
if [[ ! -f \"\$SCRIPT\" ]]; then
  echo '错误: 缺少 tests/test-reex/compare_layer_dumps_v2.py，请先从 legacy 仓库同步该脚本。' >&2
  exit 2
fi
python3 \"\$SCRIPT\" \\
  --gold '$OUT_ROOT/dump_f16_prefill${NTOK}' \\
  --test '$OUT_ROOT/dump_${TYPE}_B${B}_prefill${NTOK}' '${TYPE}_B${B}_prefill${NTOK}' \\
  --cos-threshold 0.9999 | tee '$OUT_ROOT/prefill8192_${TYPE}_B${B}_compare.log'
"
        ;;
    ppl)
        PPL_DATASET="${PPL_DATASET:-/mnt/data8t/share/datasets/model_evaluation/language_modeling/wikitext/converted/wikitext-103-raw-v1/validation.txt}"
        PPL_CTX="${PPL_CTX:-512}"
        PPL_BATCH="${PPL_BATCH:-512}"
        PPL_CHUNKS="${PPL_CHUNKS:--1}"
        run_in_container "
set -euo pipefail
cd '$WORKDIR'
export PATH=/usr/local/cuda/bin:\$PATH
export LD_LIBRARY_PATH='$BUILD_DIR'/bin:/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}
test -x '$BUILD_DIR'/bin/llama-perplexity || { echo '缺少 llama-perplexity，先 cmake --build ... --target llama-perplexity' >&2; exit 2; }
test -s '$MODEL_Q64' || { echo '缺少模型: $MODEL_Q64' >&2; exit 2; }
test -s '$PPL_DATASET' || { echo '缺少数据集: $PPL_DATASET' >&2; exit 2; }
echo \"=== PPL $TYPE B=${REEX_Q64_PSUM_BITS:-off} ngl=$NGL_TEST ctx=$PPL_CTX batch=$PPL_BATCH chunks=$PPL_CHUNKS ===\"
REEX_Q64_PSUM_BITS='${REEX_Q64_PSUM_BITS:-}' \\
  '$BUILD_DIR'/bin/llama-perplexity \\
  -m '$MODEL_Q64' \\
  -f '$PPL_DATASET' \\
  -c '$PPL_CTX' \\
  -b '$PPL_BATCH' \\
  --chunks '$PPL_CHUNKS' \\
  -ngl '$NGL_TEST'
"
        ;;
    hw-dump-smoke)
        HW_DIR="${REEX_HW_DUMP_DIR:-$OUT_ROOT/hw_vectors_smoke}"
        HW_TENSOR="${REEX_HW_DUMP_TENSOR:-ffn_moe_up}"
        HW_LAYER="${REEX_HW_DUMP_LAYER:-0}"
        run_in_container "
set -euo pipefail
cd '$WORKDIR'
export PATH=/usr/local/cuda/bin:\$PATH
export LD_LIBRARY_PATH='$BUILD_DIR'/bin:/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}
mkdir -p '$HW_DIR'
rm -rf '$HW_DIR'/*
echo \"=== hw-dump-smoke: tensor=$HW_TENSOR layer=$HW_LAYER B=8 ===\"
REEX_HW_DUMP_DIR='$HW_DIR' \\
REEX_HW_DUMP_TENSOR='$HW_TENSOR' \\
REEX_HW_DUMP_LAYER='$HW_LAYER' \\
REEX_HW_DUMP_TOKEN=0 \\
REEX_HW_DUMP_MAX_M=4 \\
REEX_Q64_PSUM_BITS=8 \\
REEX_DUMP_DIR='$HW_DIR/sw_float' REEX_DUMP_LAYER='$HW_LAYER' \\
  '$BUILD_DIR'/bin/eval_single_token -m '$MODEL_Q64' -ngl '$NGL_TEST' \\
  -p Hello -c 512 -b 512 --n-tokens 1
python3 tests/test-reex/hw_dump_postprocess.py '$HW_DIR' --expect-psum
"
        ;;
    hw-dump|hw-dump-worst|hw-dump-phase2)
        if [[ "$CMD" == "hw-dump-phase2" ]]; then
            HW_ROOT="${REEX_HW_DUMP_DIR:-$OUT_ROOT/hw_vectors_phase2_down}"
            HW_TENSOR="${REEX_HW_DUMP_TENSOR:-ffn_moe_down}"
            HW_LAYERS_STR="19 23 39"
        else
            HW_ROOT="${REEX_HW_DUMP_DIR:-$OUT_ROOT/hw_vectors_prefill8192}"
            HW_TENSOR="${REEX_HW_DUMP_TENSOR:-ffn_moe_up}"
            if [[ "$CMD" == "hw-dump-worst" ]]; then
                HW_LAYERS_STR="19 23 39"
            else
                HW_LAYERS_STR="${REEX_HW_DUMP_LAYER:-19}"
            fi
        fi
        HW_B="${REEX_Q64_PSUM_BITS:-8}"
        run_in_container "
set -euo pipefail
cd '$WORKDIR'
export PATH=/usr/local/cuda/bin:\$PATH
export LD_LIBRARY_PATH='$BUILD_DIR'/bin:/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}
mkdir -p '$HW_ROOT'
if [[ ! -s '$PROMPT_FILE' ]]; then
  echo '错误: 缺少 prompt 文件 $PROMPT_FILE，请先跑 dump 生成' >&2
  exit 2
fi
for LAYER in $HW_LAYERS_STR; do
  SW_DIR='$HW_ROOT/sw_float_L'\$LAYER
  mkdir -p \"\$SW_DIR\"
  echo \"=== hw-dump prefill8192: tensor=$HW_TENSOR layer=\$LAYER B=$HW_B ===\"
  REEX_HW_DUMP_DIR='$HW_ROOT' \\
  REEX_HW_DUMP_TENSOR='$HW_TENSOR' \\
  REEX_HW_DUMP_LAYER=\"\$LAYER\" \\
  REEX_HW_DUMP_TOKEN=0 \\
  REEX_HW_DUMP_MAX_M=4 \\
  REEX_Q64_PSUM_BITS='$HW_B' \\
  REEX_DUMP_DIR=\"\$SW_DIR\" REEX_DUMP_LAYER=\"\$LAYER\" \\
    '$BUILD_DIR'/bin/eval_single_token -m '$MODEL_Q64' -ngl '$NGL_TEST' \\
    -f '$PROMPT_FILE' -c $CTX -b $BATCH --n-tokens $NTOK
  CASE_DIR='$HW_ROOT/$HW_TENSOR-'\$LAYER
  python3 tests/test-reex/hw_dump_postprocess.py \"\$CASE_DIR\" --expect-psum
  python3 tests/test-reex/hw_dump_validate_sw.py \"\$CASE_DIR\"
done
python3 - <<PY
import json, os, time
root = '$HW_ROOT'
layers = [int(x) for x in '$HW_LAYERS_STR'.split()]
meta = {
    'phase': 2 if '$HW_TENSOR' == 'ffn_moe_down' else 1,
    'status': 'partial',
    'timestamp': int(time.time()),
    'dir': root,
    'tensor': '$HW_TENSOR',
    'prefill_tokens': $NTOK,
    'psum_bits': $HW_B,
    'layers': layers,
    'cases': []
}
for layer in layers:
    p = os.path.join(root, '$HW_TENSOR-' + str(layer), 'meta.json')
    if os.path.isfile(p):
        meta['cases'].append(json.load(open(p)))
with open(os.path.join(root, 'meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)
print('wrote', os.path.join(root, 'meta.json'))
PY
"
        ;;
    hw-validate)
        run_in_container "
set -euo pipefail
cd '$WORKDIR'
python3 tests/test-reex/hw_validate_all.py '$OUT_ROOT'
"
        ;;
    hw-export)
        run_in_container "
set -euo pipefail
cd '$WORKDIR'
python3 tests/test-reex/hw_export_rtl_bundle.py '$OUT_ROOT' --validate
echo \"=== RTL bundle ===\"
ls -lh '$OUT_ROOT/hw_vectors_rtl_bundle/manifest.json'
du -sh '$OUT_ROOT/hw_vectors_rtl_bundle'
"
        ;;
    shell)
        docker exec -it -e CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
            "$CONTAINER" bash -lc "cd '$WORKDIR' && exec bash"
        ;;
    *)
        usage
        exit 2
        ;;
esac

