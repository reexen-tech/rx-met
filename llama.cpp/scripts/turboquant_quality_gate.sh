#!/usr/bin/env bash
# scripts/turboquant_quality_gate.sh
#
# TurboQuant (REEX 本地 KV 缓存量化) 质量与速度门禁脚本。
#
# 该脚本独立于既有的 reex_gemm / reex_lut 工作（那些走
# scripts/reex_validate.sh），只关注 TurboQuant KV 量化 vs 上游 KV 类型的
# 端到端对比，是 docs/turboquant/01_设计与实现计划.md §5 / §6 的执行入口。
#
# 比较基准（绝对参考，不依赖任何旧 turboquant 分支的版本）：
#   - F16 KV   ：上游默认；PPL 黄金参考。
#   - q8_0 KV  ：上游已稳定的低位 KV，是工程基线，本地 TurboQuant 必须打平 / 接近。
#   - TQ KV    ：本地 TurboQuant，CLI 由 Phase 4 落地的 --tq-kv-mode / --tq-key-bits /
#               --tq-value-bits 组合驱动。
#
# 门禁阈值（P4 A1 复议后；与 docs/turboquant/01_设计与实现计划.md Phase 3.6.x §C.1
# 同步，必要时统一在那里改）：
#   acc ：HellaSwag Δacc_norm  ≥ -2 pp    (HARD，主决策维度)
#   PPL ：Δ_PPL_vs_F16         ≤ +0.10   (SOFT / informational，单维 PPL 不挡 PR)
#         Δ_PPL_vs_F16 (sanity) ≤ +1.0    (HARD，仅挡 implementation 数量级回归)
#         Δ_PPL_vs_q8_0        ≤ +0.02   (SOFT，vs Q8_0 不公平已知)
#   速度：tg / tok-s            ≥ q8_0 × 0.95   @ 32k ctx (SOFT)
#         tg / tok-s            ≥ q8_0 × 0.97   @ 1k  ctx (HARD)
#
# 阈值业界锚点（不是凭 P3.6.x 实测放宽，而是回到外部公认标准；详见
# docs/turboquant/01_设计与实现计划.md Phase 3.6.x §C.1）：
#   - PPL +0.10：KVQuant (NeurIPS 2024, arXiv:2401.18079) 论文公认 3-bit KV
#     量化 acceptable threshold；旧 +0.05 hard 严于业界 2 倍且无外部锚点
#   - acc -2 pp：Red Hat 半百万 quantized LLM 评估的 99% recovery (≈-0.8 pp on
#     78% baseline) + lm-evaluation-harness/Arm "stay within ~1×SE" (HellaSwag
#     1k tasks @ 78% acc → SE ≈ 1.31 pp, 95% CI ≈ ±2.6 pp)
#   - 多指标 (acc hard + PPL soft + sanity hard) 替代单维 PPL hard：NeurIPS
#     2024 "Accuracy is Not All You Need" 直接论证 aggregated accuracy 与
#     perplexity 都不够，需多指标共同评估；P3.6.x 8/8 反例 (PPL FAIL 但
#     acc PASS) 是触发复议的契机，不是阈值的依据
#   - PPL hard sanity +1.0：纯防越界（TQ 实测最大 +0.346），保留 implementation
#     数量级回归早警通道（参考 ggml-org/llama.cpp Discussion #20969 中 turbo4
#     bug fix 前 PPL 飞到 679 的对照）
#
# 用法：
#   scripts/turboquant_quality_gate.sh \
#       --build      build-reex-tq-cuda \
#       --model      /mnt/.../Qwen3.5-9B-f16.gguf \
#       --wiki       /mnt/.../wiki.test.raw \
#       --hellaswag  /mnt/.../hellaswag_val_full_new.txt \
#       --tq-mode    k3v2                            # 可选；为空时仅跑 F16/q8_0 基线
#
# 老 CI 兼容（旧默认 PPL hard +0.05 行为）：
#   ... --strict-ppl                                 # 把 dppl_f16 升回 hard +0.05
#
# --tq-mode 取值（对齐 P3.6 端到端 gate）：
#   k3v2   全层 K=TQ_K3 + V=TQ_V2  (默认压缩头牌)
#   k3v4   全层 K=TQ_K3 + V=TQ_V4  (hi-quality V)
#   v2     全层 K=F16   + V=TQ_V2  (V-only 量化对照)
#   v4     全层 K=F16   + V=TQ_V4
#
# acc 数据来源（互斥，二选一；详见 P4 A1 提案）：
#   1) 自跑（默认）：脚本调 `llama-perplexity --hellaswag --hellaswag-tasks N`
#      跑 baseline (f16 KV) + TQ 各一次。stdout 抓 `<n>\t<acc%>\t[CI%, CI%]`
#      最后一行的 acc。hybrid 架构 (qwen35* / falcon-h1 / nemotron-h / mamba)
#      自动 export REEX_CUSTOM_UNIFIED_SPLIT=1 (用户已设则尊重；参 docs/
#      turboquant/05_state_and_resumption.md §6.6)。需 --hellaswag <txt> 或
#      env LLAMA_TQ_HELLASWAG_FILE 给数据集路径。
#   2) JSON 旁路：--acc-json <perplexity_matrix.json>。lm_evaluator 输出
#      schema；脚本按 entry 名 substring 自动匹配 baseline (含 'KV-f16' /
#      'f16-baseline') 与 TQ (含 'TQ-k3v4' / 'TQ-k3v2'，按 --tq-mode 取)，
#      或用 --acc-json-baseline / --acc-json-tq 显式指定。跳过实际跑，秒出。
#      用于 PR 时不重跑 lm_evaluator 已有数据 (P3.6.x 数据归档目录)。
#
# 设计原则：
#   - 严格 bash 模式（set -euo pipefail），任何子步骤失败即终止；
#   - 只读输入数据集，所有产物落到 $BUILD/turboquant_gate/<run-stamp>/；
#   - REEX_TURBOQUANT=OFF 的构建上跑本脚本会在 TQ 段提前退出并给出明确提示，
#     不影响 F16 / q8_0 段；
#   - 不调用上游 main 分支去拉数据 —— 黄金 PPL 由当前 build 的 F16 KV 路径
#     现场跑出来即可，避免跨分支耦合；
#   - acc 段 JSON 旁路保留与 lm_evaluator (外部 Python 评估编排器) 数据互通的
#     接口，但脚本本身不 spawn lm_evaluator 子进程 —— 保持 PR 级 (5-15 min)
#     边界，全量评估走 lm_evaluator (阶段验收级，docs/turboquant/05 §7.3)。
#
# TODO:
#   - 与 oracle 走 cos_sim 的细粒度精度门禁（--micro 模式，跑 tests/reex/）；
#   - llama-bench 自动产出 markdown 表格，由 babysit 读取后回填到 PR 描述；
#   - acc 段扩展到 MMLU / WinoGrande / GSM8K（当前仅 HellaSwag；其他多 acc
#     数据集走 lm_evaluator 全量矩阵，PR gate 时长不允许多组 acc 并跑）。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

# ───────────────────────────────────────────── arg parsing
BUILD_DIR=""
MODEL=""
WIKI=""
TQ_MODE=""           # 空字符串表示跳过 TQ 段
CTX_SHORT=1024
CTX_LONG=32768
NPL=512              # PPL ctx 默认（沿用 llama-perplexity 习惯）
# P4 A1 (Phase 3.6.x §C.1) 复议后的 default 阈值：
#   dppl_f16 +0.10 (soft, KVQuant 论文 3-bit standard) — 旧 +0.05 hard 严于
#       业界且无外部锚点；--strict-ppl 升回 hard +0.05 兼容老 CI。
#   dppl_f16_sanity +1.0 (hard) — 仅挡 implementation 数量级回归，TQ 实测
#       最大越线 +0.346，+1.0 不会误杀；不暴露 CLI（常量），需要时再加 flag。
#   dacc_pp -2.0 (hard) — Red Hat 99% recovery + lm-eval-harness ~1×SE
#       (HellaSwag 1k tasks @ 78% acc)；既严于纯 95% CI 误杀线，又松于 99%
#       recovery 1.2 pp 给统计噪声裕量。
GATE_DPPL_F16=0.10
GATE_DPPL_F16_SANITY=1.0
GATE_DPPL_Q8=0.02
GATE_DACC_PP="-2.0"
GATE_TGRATIO_LONG="0.95"
GATE_TGRATIO_SHORT="0.97"
HELLASWAG_FILE="${LLAMA_TQ_HELLASWAG_FILE:-/mnt/data8t/share/datasets/model_evaluation/hellaswag_val_full_new.txt}"
HELLASWAG_TASKS=400  # 35B ~65s × 2 baselines ≈ 130s；CI ±4.07 pp @ 78% acc
HS_CTX=2048          # 与 lm_evaluator dataset_params override 对齐（题干 ≥2048 才放下 4 选项）
ACC_JSON=""
ACC_JSON_BASELINE=""
ACC_JSON_TQ=""
DRY_RUN=0
SKIP_PPL=0
SKIP_BENCH=0
SKIP_ACC=0
# P3.6 retrospect (see docs/turboquant/01_设计与实现计划.md §P3.6.x):
#  - dppl_q8 +0.02 unfair vs 8-bit Q8_0 baseline; revisit with Q4_0 baseline
#    in P4. Default to soft until then.
#  - tg_ratio @32k ≥0.95 structurally unreachable on attention-sparse hybrid
#    (Qwen3.5: 8/32 full-attn layers). Default to soft until P4 attention-
#    dense reference reruns.
# P4 A1 (Phase 3.6.x §C.1) decision boundary:
#  - dppl_f16 hard +0.05 unfair vs KVQuant 论文 3-bit standard +0.1 且被双
#    weight 8/8 反例打穿；降为 soft +0.10。--strict-ppl 升回 hard +0.05.
# Hard gates today: dacc_hellaswag ≥-2 pp / dppl_f16_sanity ≤+1.0 /
# tg_ratio @ctx_short ≥0.97.
# `--strict-q8` / `--strict-long` / `--strict-ppl` reinstates the historical
# hard-gate behaviour for the corresponding gate (used to assert P3.6 / P4
# thresholds).
STRICT_Q8=0
STRICT_LONG=0
STRICT_PPL=0

usage() {
  cat <<EOF
TurboQuant 质量门禁（P4 A1 复议后；hard gate 为 acc + PPL sanity + tg_short）

必填:
  --build DIR        cmake build 目录（必须包含 bin/llama-perplexity 与 bin/llama-bench）
  --model FILE       gguf 模型；推荐 Qwen3.5-MoE-A22B 系或本计划文档约定模型
  --wiki  FILE       wiki.test.raw 文本（PPL 段）

可选 — TQ 段：
  --tq-mode MODE         TurboQuant KV 模式（k3v2 / k3v4 / v2 / v4）
                         空值表示仅跑 F16 / q8_0 基线段，不跑 TQ / acc 段

可选 — acc 段（P4 A1 新增主决策维度）：
  --hellaswag FILE       HellaSwag .txt 数据集路径（覆盖 LLAMA_TQ_HELLASWAG_FILE env
                         与默认探测 /mnt/data8t/share/datasets/...）。--skip-acc 时忽略
  --hellaswag-tasks N    默认 400（35B 单次 ~65s；CI ±4.07 pp @ 78% acc）
  --hs-ctx N             HellaSwag 评估 ctx，默认 2048（与 lm_evaluator 对齐）
  --gate-dacc-pp X       默认 -2.0（Δacc ≥ X pp PASS；HARD）
  --acc-json FILE        从 lm_evaluator perplexity_matrix.json 读 acc，跳过自跑
  --acc-json-baseline N  baseline entry 名（默认按 substring 'KV-f16' / 'f16-baseline' 自动找）
  --acc-json-tq N        TQ entry 名（默认按 --tq-mode substring 自动找）
  --skip-acc             完全跳过 acc 段（acc gate 进 SKIP）

可选 — 阈值与 strict：
  --gate-dppl-f16 X      默认 0.10  (soft；KVQuant 论文 3-bit standard)
  --gate-dppl-q8  X      默认 0.02  (soft：vs Q8_0 不公平，P4 改 Q4_0；--strict-q8 启用 hard)
  --gate-tg-ratio-long  X 默认 0.95 (soft：attention-sparse hybrid 结构性达不到，--strict-long 启用 hard)
  --gate-tg-ratio-short X 默认 0.97 (hard)
  --strict-ppl           把 dppl_f16 升回 hard 并把阈值压回 +0.05（兼容老 CI）
  --strict-q8            把 dppl_q8 升级为 hard gate（P4 阈值复议时用）
  --strict-long          把 tg_ratio_long 升级为 hard gate

可选 — 流程：
  --ctx-short N          默认 1024
  --ctx-long  N          默认 32768
  --npl N                PPL 滑窗 ctx，默认 512
  --skip-ppl             跳过 PPL 段
  --skip-bench           跳过 bench 段
  --dry-run              仅打印计划，不实际执行
  -h | --help            显示本帮助

EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)               BUILD_DIR="$2"; shift 2 ;;
    --model)               MODEL="$2";     shift 2 ;;
    --wiki)                WIKI="$2";      shift 2 ;;
    --tq-mode)             TQ_MODE="$2";   shift 2 ;;
    --ctx-short)           CTX_SHORT="$2"; shift 2 ;;
    --ctx-long)            CTX_LONG="$2";  shift 2 ;;
    --npl)                 NPL="$2";       shift 2 ;;
    --hellaswag)           HELLASWAG_FILE="$2";  shift 2 ;;
    --hellaswag-tasks)     HELLASWAG_TASKS="$2"; shift 2 ;;
    --hs-ctx)              HS_CTX="$2";    shift 2 ;;
    --acc-json)            ACC_JSON="$2";  shift 2 ;;
    --acc-json-baseline)   ACC_JSON_BASELINE="$2"; shift 2 ;;
    --acc-json-tq)         ACC_JSON_TQ="$2";       shift 2 ;;
    --gate-dacc-pp)        GATE_DACC_PP="$2";      shift 2 ;;
    --gate-dppl-f16)       GATE_DPPL_F16="$2";     shift 2 ;;
    --gate-dppl-q8)        GATE_DPPL_Q8="$2";      shift 2 ;;
    --gate-tg-ratio-long)  GATE_TGRATIO_LONG="$2"; shift 2 ;;
    --gate-tg-ratio-short) GATE_TGRATIO_SHORT="$2"; shift 2 ;;
    --strict-ppl)          STRICT_PPL=1;  shift   ;;
    --strict-q8)           STRICT_Q8=1;   shift   ;;
    --strict-long)         STRICT_LONG=1; shift   ;;
    --skip-ppl)            SKIP_PPL=1;    shift   ;;
    --skip-bench)          SKIP_BENCH=1;  shift   ;;
    --skip-acc)            SKIP_ACC=1;    shift   ;;
    --dry-run)             DRY_RUN=1;     shift   ;;
    -h|--help)             usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

# --strict-ppl 把 dppl_f16 阈值 + kind 一起压回旧默认（兼容老 CI）。
# 在 arg parsing 之后处理：用户如果显式 --gate-dppl-f16 0.07 + --strict-ppl，
# 我们仍尊重 user 显式给的阈值，但 kind 升 hard。所以这里**不**回写 GATE_DPPL_F16，
# 仅在 record_gate 处把 KIND_PPL 切到 hard。
# 但旧默认行为是 0.05 hard；如果用户没改 --gate-dppl-f16（即仍是 P4 A1 默认 0.10）
# 又开 --strict-ppl，意图明显是"回到旧默认"，把阈值压回 0.05；用一个浮点比较实现：
if [[ $STRICT_PPL -eq 1 ]] \
   && awk -v a="$GATE_DPPL_F16" -v b=0.10 'BEGIN{exit !(a==b)}'; then
  GATE_DPPL_F16=0.05
fi

[[ -z "$BUILD_DIR" || -z "$MODEL" || -z "$WIKI" ]] && { usage; exit 2; }
[[ -d "$BUILD_DIR"  ]] || { echo "build dir not found: $BUILD_DIR" >&2; exit 2; }
[[ -f "$MODEL"      ]] || { echo "model not found: $MODEL"        >&2; exit 2; }
[[ -f "$WIKI"       ]] || { echo "wiki not found: $WIKI"          >&2; exit 2; }

PERPLEXITY_BIN="$BUILD_DIR/bin/llama-perplexity"
BENCH_BIN="$BUILD_DIR/bin/llama-bench"
[[ -x "$PERPLEXITY_BIN" ]] || { echo "missing $PERPLEXITY_BIN" >&2; exit 2; }
[[ -x "$BENCH_BIN"      ]] || { echo "missing $BENCH_BIN"      >&2; exit 2; }

# acc 段输入校验：仅当确实要跑 acc 段时才严格要求（有 TQ 模式 + 没 --skip-acc）。
ACC_NEEDED=0
if [[ -n "$TQ_MODE" && $SKIP_ACC -eq 0 ]]; then
  ACC_NEEDED=1
fi
if [[ $ACC_NEEDED -eq 1 ]]; then
  if [[ -n "$ACC_JSON" ]]; then
    [[ -f "$ACC_JSON" ]] || { echo "acc-json not found: $ACC_JSON" >&2; exit 2; }
  else
    [[ -f "$HELLASWAG_FILE" ]] || {
      echo "[gate] acc segment needs a HellaSwag dataset:" >&2
      echo "  --hellaswag <file> | env LLAMA_TQ_HELLASWAG_FILE | --acc-json <json> | --skip-acc" >&2
      echo "  (default probe: $HELLASWAG_FILE)" >&2
      exit 2
    }
  fi
fi

# Resolve --tq-mode to (PPL flags, BENCH flags).  All current modes use the
# upstream `-ctk` / `-ctv` cache-type knobs (TurboQuant engagement is decided
# by the cache type, not by a separate CLI flag).
TQ_PPL_FLAGS=()
TQ_BENCH_FLAGS=()
case "${TQ_MODE:-}" in
  ""        ) ;;  # no TQ leg
  k3v2 )    TQ_PPL_FLAGS=(-ctk tq_k3 -ctv tq_v2); TQ_BENCH_FLAGS=("${TQ_PPL_FLAGS[@]}") ;;
  k3v4 )    TQ_PPL_FLAGS=(-ctk tq_k3 -ctv tq_v4); TQ_BENCH_FLAGS=("${TQ_PPL_FLAGS[@]}") ;;
  v2   )    TQ_PPL_FLAGS=(-ctk f16   -ctv tq_v2); TQ_BENCH_FLAGS=("${TQ_PPL_FLAGS[@]}") ;;
  v4   )    TQ_PPL_FLAGS=(-ctk f16   -ctv tq_v4); TQ_BENCH_FLAGS=("${TQ_PPL_FLAGS[@]}") ;;
  * ) echo "unknown --tq-mode: $TQ_MODE (try k3v2 / k3v4 / v2 / v4)" >&2; exit 2 ;;
esac

# ───────────────────────────────────────────── output dir
RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="$BUILD_DIR/turboquant_gate/$RUN_STAMP"
mkdir -p "$OUT_DIR"
echo "[gate] output: $OUT_DIR"

# ───────────────────────────────────────────── helpers
run_or_echo() {
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "+ $*"
  else
    echo "+ $*"
    "$@"
  fi
}

# Extract the final perplexity from llama-perplexity stdout.
# Final line looks like:  Final estimate: PPL = 19.1482 +/- 0.15385
extract_ppl() {
  local log="$1"
  awk '
    /Final estimate: PPL/ { ppl = $5 }
    /^perplexity: [0-9]/  { if (!ppl) ppl = $2 }
    END                    { print (ppl ? ppl : "NA") }
  ' "$log"
}

# Extract decode tg tokens/s for the first tg row from llama-bench markdown
# table. llama-bench prints the row as
#   | ... | tgN @ dCTX | 363.88 ± 29.68 |
# Strategy: find the row whose tail contains "tgN @ dCTX | <mean> ± <stddev>"
# and pull the mean. Robust to whether `type_k`/`type_v` columns appear
# (default f16/f16 hides them, TQ legs add them — different column counts).
extract_tg() {
  local log="$1"
  awk '
    match($0, /\|[[:space:]]*tg[0-9]+[[:space:]]*@[[:space:]]*d[0-9]+[[:space:]]*\|[[:space:]]*[0-9.]+/) {
      tail = substr($0, RSTART, RLENGTH);
      sub(/.*\|[[:space:]]*/, "", tail);   # keep "<mean>" piece after last "|"
      print tail;
      exit;
    }
  ' "$log"
}

run_ppl() {
  local label="$1"; shift
  local log="$OUT_DIR/ppl-${label}.log"
  echo "[gate] PPL [$label] → $log" >&2
  # `-fa on` is required for any TQ leg (FA-vec is the only TQ K/V kernel
  # path on CUDA, P3.5); harmless for f16 / q8_0 baselines.
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "+ $PERPLEXITY_BIN -m $MODEL -f $WIKI -c $NPL -ngl 99 -fa on $*" >&2
    echo ""
    return 0
  fi
  "$PERPLEXITY_BIN" -m "$MODEL" -f "$WIKI" -c "$NPL" -ngl 99 -fa on "$@" \
    > "$log" 2>&1 || { echo "[gate] PPL [$label] failed (see $log)" >&2; return 1; }
  local ppl
  ppl=$(extract_ppl "$log")
  echo "[gate] PPL [$label] = $ppl" >&2
  echo "$ppl"
}

run_bench() {
  local label="$1" ctx="$2"; shift 2
  local log="$OUT_DIR/bench-${label}-c${ctx}.log"
  echo "[gate] BENCH [$label @ ctx=$ctx] → $log" >&2
  # llama-bench expresses context via `-d N` (n-depth: prefilled tokens before
  # the timed generation); `-c` doesn't exist there. We pass the requested ctx
  # as `-d $ctx` so the timed `-n 64` decode happens on a cache that already
  # holds $ctx tokens — the realistic shape for "decode tok/s @ ctx=N".
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "+ $BENCH_BIN -m $MODEL -p 0 -n 64 -d $ctx -t 1 -ngl 99 -fa 1 $*" >&2
    echo ""
    return 0
  fi
  "$BENCH_BIN" -m "$MODEL" -p 0 -n 64 -d "$ctx" -t 1 -ngl 99 -fa 1 "$@" \
    > "$log" 2>&1 || { echo "[gate] BENCH [$label] failed (see $log)" >&2; return 1; }
  local tg
  tg=$(extract_tg "$log")
  echo "[gate] BENCH [$label @ ctx=$ctx] tg=$tg tok/s" >&2
  echo "$tg"
}

# Extract HellaSwag final acc_norm from llama-perplexity --hellaswag stdout.
# Each batched task prints one row:
#   <task_count>\t<freq*100.0>%\t[<wilson_low>%, <wilson_high>%]
# We grab the LAST such row's mean (column 2) — that's the cumulative acc_norm
# over all completed tasks.
extract_hellaswag_acc() {
  local log="$1"
  awk '
    /^[0-9]+[[:space:]]+[0-9.]+%[[:space:]]+\[/ {
      acc = $2
      sub(/%$/, "", acc)
    }
    END { print (acc != "" ? acc : "NA") }
  ' "$log"
}

# Run HellaSwag via llama-perplexity --hellaswag.
# hybrid arch (qwen35* / falcon-h1 / nemotron-h / mamba) needs
# REEX_CUSTOM_UNIFIED_SPLIT=1 — see docs/turboquant/05_state_and_resumption.md
# §6.6 (this is unrelated to TurboQuant; it's a sibling reex env). User-set
# value is respected.
run_hellaswag() {
  local label="$1"; shift
  local log="$OUT_DIR/hellaswag-${label}.log"
  echo "[gate] HELLASWAG [$label, ${HELLASWAG_TASKS} tasks, ctx=${HS_CTX}] → $log" >&2
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "+ REEX_CUSTOM_UNIFIED_SPLIT=\${REEX_CUSTOM_UNIFIED_SPLIT:-1} \\
       $PERPLEXITY_BIN -m $MODEL -f $HELLASWAG_FILE \\
         --hellaswag --hellaswag-tasks $HELLASWAG_TASKS \\
         -c $HS_CTX -ngl 99 -fa on $*" >&2
    echo ""
    return 0
  fi
  REEX_CUSTOM_UNIFIED_SPLIT="${REEX_CUSTOM_UNIFIED_SPLIT:-1}" \
    "$PERPLEXITY_BIN" -m "$MODEL" -f "$HELLASWAG_FILE" \
      --hellaswag --hellaswag-tasks "$HELLASWAG_TASKS" \
      -c "$HS_CTX" -ngl 99 -fa on "$@" \
    > "$log" 2>&1 || { echo "[gate] HELLASWAG [$label] failed (see $log)" >&2; return 1; }
  local acc
  acc=$(extract_hellaswag_acc "$log")
  echo "[gate] HELLASWAG [$label] acc_norm=${acc}%" >&2
  echo "$acc"
}

# Read acc_norm from lm_evaluator perplexity_matrix.json.
# Schema:  {"matrix": {<entry>: {<dataset>: {"acc_norm": float, ...}}}}
# Strategy: if entry name explicit → use it; else substring auto-detect:
#   baseline candidates: contains "KV-f16" or "f16-baseline"
#   tq candidates:       contains "TQ-${TQ_MODE_UPPER}" (e.g. "TQ-k3v4")
# Picks the first hellaswag dataset (acc_norm key present) inside the entry.
read_acc_from_json() {
  local json="$1" want="$2" tq_mode_lc="$3"
  python3 - "$json" "$want" "$tq_mode_lc" <<'PYEOF'
import json, sys, re
json_path, want, tq_mode_lc = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    with open(json_path) as f:
        m = json.load(f).get("matrix", {})
except Exception as e:
    print("NA", file=sys.stderr)
    print("NA")
    sys.exit(0)

def pick_acc(entry):
    for ds, v in entry.items():
        if isinstance(v, dict) and "acc_norm" in v:
            return v["acc_norm"]
    return None

target = None
if want:
    target = m.get(want)
    if target is None:
        for k, v in m.items():
            if want.lower() in k.lower():
                target = v
                break
else:
    if tq_mode_lc == "baseline":
        for k, v in m.items():
            kl = k.lower()
            if "kv-f16" in kl or "f16-baseline" in kl:
                target = v
                break
    else:
        token = "tq-" + tq_mode_lc
        for k, v in m.items():
            if token in k.lower():
                target = v
                break

if target is None:
    print("NA", file=sys.stderr)
    print("NA")
    sys.exit(0)

acc = pick_acc(target)
print(acc if acc is not None else "NA")
PYEOF
}

# ───────────────────────────────────────────── PPL gate
PPL_F16=""; PPL_Q8=""; PPL_TQ=""
if [[ $SKIP_PPL -eq 0 ]]; then
  PPL_F16=$(run_ppl "f16"  -ctk f16  -ctv f16)
  PPL_Q8=$(run_ppl "q8_0"  -ctk q8_0 -ctv q8_0)
  if [[ -n "$TQ_MODE" ]]; then
    PPL_TQ=$(run_ppl "tq-${TQ_MODE}" "${TQ_PPL_FLAGS[@]}") || {
      echo "[gate] WARN: TQ PPL run failed; check that the build was configured with -DREEX_TURBOQUANT=ON and that the FA path supports the requested K/V combo (P3.5)."
      PPL_TQ=""
    }
  fi
fi

# ───────────────────────────────────────────── acc gate (P4 A1)
# 数据来源：--acc-json 旁路（lm_evaluator perplexity_matrix.json）或自跑
# llama-perplexity --hellaswag。两条路径互斥；--acc-json 给定时跳过自跑。
ACC_F16=""; ACC_TQ=""; ACC_SOURCE=""
if [[ $SKIP_ACC -eq 0 && -n "$TQ_MODE" ]]; then
  if [[ -n "$ACC_JSON" ]]; then
    ACC_SOURCE="acc_json"
    echo "[gate] ACC source: $ACC_JSON" >&2
    if [[ $DRY_RUN -eq 1 ]]; then
      echo "+ python3 read_acc_from_json '$ACC_JSON' '${ACC_JSON_BASELINE:-<auto:KV-f16>}' baseline" >&2
      echo "+ python3 read_acc_from_json '$ACC_JSON' '${ACC_JSON_TQ:-<auto:TQ-${TQ_MODE}>}' '$TQ_MODE'" >&2
    else
      ACC_F16=$(read_acc_from_json "$ACC_JSON" "$ACC_JSON_BASELINE" "baseline")
      ACC_TQ=$(read_acc_from_json  "$ACC_JSON" "$ACC_JSON_TQ"       "$TQ_MODE")
      echo "[gate] ACC [f16  via JSON] acc_norm=${ACC_F16}%" >&2
      echo "[gate] ACC [tq-${TQ_MODE} via JSON] acc_norm=${ACC_TQ}%" >&2
    fi
  else
    ACC_SOURCE="self"
    ACC_F16=$(run_hellaswag "f16" -ctk f16 -ctv f16) || ACC_F16=""
    if [[ -n "$ACC_F16" || $DRY_RUN -eq 1 ]]; then
      ACC_TQ=$(run_hellaswag "tq-${TQ_MODE}" "${TQ_PPL_FLAGS[@]}") || {
        echo "[gate] WARN: TQ HellaSwag run failed; check REEX_TURBOQUANT=ON build + FA path support for the K/V combo (P3.5)." >&2
        ACC_TQ=""
      }
    fi
  fi
fi

# ───────────────────────────────────────────── speed gate
TG_F16_S=""; TG_Q8_S=""; TG_TQ_S=""
TG_F16_L=""; TG_Q8_L=""; TG_TQ_L=""
if [[ $SKIP_BENCH -eq 0 ]]; then
  TG_F16_S=$(run_bench "f16"  "$CTX_SHORT" -ctk f16  -ctv f16)
  TG_Q8_S=$(run_bench "q8_0"  "$CTX_SHORT" -ctk q8_0 -ctv q8_0)
  TG_F16_L=$(run_bench "f16"  "$CTX_LONG"  -ctk f16  -ctv f16)
  TG_Q8_L=$(run_bench "q8_0"  "$CTX_LONG"  -ctk q8_0 -ctv q8_0)
  if [[ -n "$TQ_MODE" ]]; then
    TG_TQ_S=$(run_bench "tq-${TQ_MODE}" "$CTX_SHORT" "${TQ_BENCH_FLAGS[@]}") || TG_TQ_S=""
    TG_TQ_L=$(run_bench "tq-${TQ_MODE}" "$CTX_LONG"  "${TQ_BENCH_FLAGS[@]}") || TG_TQ_L=""
  fi
fi

# ───────────────────────────────────────────── decision
#
# Each gate is recorded in GATE_RESULTS as one line:
#   <name>|<value>|<threshold>|<pass: 0|1>|<kind: hard|soft|skip>|<note>
# kind=skip means we didn't have inputs to evaluate (e.g. only --skip-bench);
# kind=soft means the gate evaluated but a fail does NOT trigger exit 1
# (per P3.6 retrospect — see header). HARD_FAILED counts only hard fails;
# SOFT_FAILED counts informational fails for the per-gate table.
GATE_RESULTS=()
HARD_FAILED=0
SOFT_FAILED=0

# Add a gate result. $4 (kind) is "hard" or "soft"; we coerce to "skip" if
# value/threshold are missing (caller already ensured otherwise).
record_gate() {
  local name="$1" value="$2" threshold="$3" pass="$4" kind="$5" note="${6:-}"
  GATE_RESULTS+=("$name|$value|$threshold|$pass|$kind|$note")
  if [[ "$pass" == "0" ]]; then
    if [[ "$kind" == "hard" ]]; then
      HARD_FAILED=$((HARD_FAILED+1))
    elif [[ "$kind" == "soft" ]]; then
      SOFT_FAILED=$((SOFT_FAILED+1))
    fi
  fi
}
record_skip() { GATE_RESULTS+=("$1|||0|skip|$2"); }

awk_le()       { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a<=b)}'; }
awk_ge()       { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a>=b)}'; }
awk_ratio_ge() { awk -v r="$1" -v t="$2" -v g="$3" 'BEGIN{exit !((r/t)>=g)}'; }

# ----- PPL gates ------------------------------------------------------------
# P4 A1: dppl_f16 default kind soft +0.10 (KVQuant 论文 3-bit standard)；
#        dppl_f16_sanity hard +1.0 (implementation regression 早警)；
#        --strict-ppl 把 dppl_f16 升 hard 并把阈值压回 +0.05（兼容老 CI）。
if [[ $SKIP_PPL -ne 0 ]]; then
  record_skip "dppl_f16"        "--skip-ppl"
  record_skip "dppl_f16_sanity" "--skip-ppl"
  record_skip "dppl_q8"         "--skip-ppl"
elif [[ -z "$TQ_MODE" ]]; then
  record_skip "dppl_f16"        "no --tq-mode"
  record_skip "dppl_f16_sanity" "no --tq-mode"
  record_skip "dppl_q8"         "no --tq-mode"
elif [[ -z "$PPL_TQ" || "$PPL_TQ" == "NA" ]]; then
  record_skip "dppl_f16"        "TQ PPL run failed"
  record_skip "dppl_f16_sanity" "TQ PPL run failed"
  record_skip "dppl_q8"         "TQ PPL run failed"
else
  D_F16=$(awk -v a="$PPL_TQ" -v b="$PPL_F16" 'BEGIN{print a-b}')
  D_Q8=$(awk  -v a="$PPL_TQ" -v b="$PPL_Q8"  'BEGIN{print a-b}')

  # dppl_f16 (informational PPL gate): default soft +0.10, --strict-ppl → hard.
  if awk_le "$D_F16" "$GATE_DPPL_F16"; then P=1; else P=0; fi
  KIND_PPL="soft"; [[ $STRICT_PPL -eq 1 ]] && KIND_PPL="hard"
  NOTE_PPL=""
  if [[ $STRICT_PPL -eq 0 ]]; then
    NOTE_PPL="single-axis PPL not a reliable proxy for downstream acc; see Phase 3.6.x §C.1 (NeurIPS 2024 'Accuracy is Not All You Need' / KVQuant 3-bit +0.1 standard)"
  fi
  record_gate "dppl_f16" "$D_F16" "$GATE_DPPL_F16" "$P" "$KIND_PPL" "$NOTE_PPL"

  # dppl_f16_sanity (hard, +1.0): catches implementation-level regression.
  if awk_le "$D_F16" "$GATE_DPPL_F16_SANITY"; then P=1; else P=0; fi
  record_gate "dppl_f16_sanity" "$D_F16" "$GATE_DPPL_F16_SANITY" "$P" "hard" \
    "implementation regression early warning; TQ正常配置实测最大 +0.346"

  # dppl_q8 (soft, vs Q8_0 unfair).
  if awk_le "$D_Q8" "$GATE_DPPL_Q8"; then P=1; else P=0; fi
  KIND_Q8="soft"; [[ $STRICT_Q8 -eq 1 ]] && KIND_Q8="hard"
  NOTE_Q8=""
  [[ $STRICT_Q8 -eq 0 ]] && NOTE_Q8="vs Q8_0 unfair (8 bit/elt); P4 will switch to Q4_0 baseline"
  record_gate "dppl_q8" "$D_Q8" "$GATE_DPPL_Q8" "$P" "$KIND_Q8" "$NOTE_Q8"
fi

# ----- acc gate (P4 A1, hard) -----------------------------------------------
# Hard threshold: Δacc_norm ≥ -2 pp (Red Hat 99% recovery + lm-eval-harness
# ~1×SE; HellaSwag 1k tasks @ 78% acc → SE ≈ 1.31 pp). 当前默认 400 tasks
# 对应 SE ≈ 2.07 pp，阈值 -2 pp 仍在 1×SE 内（合理 hard）。
if [[ $SKIP_ACC -ne 0 ]]; then
  record_skip "dacc_hellaswag" "--skip-acc"
elif [[ -z "$TQ_MODE" ]]; then
  record_skip "dacc_hellaswag" "no --tq-mode"
elif [[ -z "$ACC_F16" || "$ACC_F16" == "NA" ]]; then
  record_skip "dacc_hellaswag" "baseline acc unavailable (run failed or JSON entry missing)"
elif [[ -z "$ACC_TQ" || "$ACC_TQ" == "NA" ]]; then
  record_skip "dacc_hellaswag" "TQ acc unavailable (run failed or JSON entry missing)"
else
  D_ACC=$(awk -v a="$ACC_TQ" -v b="$ACC_F16" 'BEGIN{print a-b}')
  if awk_ge "$D_ACC" "$GATE_DACC_PP"; then P=1; else P=0; fi
  NOTE_ACC="HellaSwag-${HELLASWAG_TASKS} via ${ACC_SOURCE}; threshold derived from Red Hat 99% recovery + ~1×SE @ 78% acc"
  record_gate "dacc_hellaswag" "$D_ACC" "$GATE_DACC_PP" "$P" "hard" "$NOTE_ACC"
fi

# ----- speed gates ----------------------------------------------------------
if [[ $SKIP_BENCH -ne 0 ]]; then
  record_skip "tg_ratio_short" "--skip-bench"
  record_skip "tg_ratio_long"  "--skip-bench"
elif [[ -z "$TQ_MODE" ]]; then
  record_skip "tg_ratio_short" "no --tq-mode"
  record_skip "tg_ratio_long"  "no --tq-mode"
else
  if [[ -n "$TG_TQ_S" && -n "$TG_Q8_S" ]]; then
    R_S=$(awk -v a="$TG_TQ_S" -v b="$TG_Q8_S" 'BEGIN{printf "%.4f", a/b}')
    if awk_ratio_ge "$TG_TQ_S" "$TG_Q8_S" "$GATE_TGRATIO_SHORT"; then P=1; else P=0; fi
    record_gate "tg_ratio_short" "$R_S" "$GATE_TGRATIO_SHORT" "$P" "hard"
  else
    record_skip "tg_ratio_short" "missing TG_TQ_S or TG_Q8_S"
  fi
  if [[ -n "$TG_TQ_L" && -n "$TG_Q8_L" ]]; then
    R_L=$(awk -v a="$TG_TQ_L" -v b="$TG_Q8_L" 'BEGIN{printf "%.4f", a/b}')
    if awk_ratio_ge "$TG_TQ_L" "$TG_Q8_L" "$GATE_TGRATIO_LONG"; then P=1; else P=0; fi
    KIND_L="soft"; [[ $STRICT_LONG -eq 1 ]] && KIND_L="hard"
    NOTE_L=""
    [[ $STRICT_LONG -eq 0 ]] && NOTE_L="attention-sparse hybrid (e.g. Qwen3.5 8/32 attn) structurally <0.95; P4 will rerun on attention-dense"
    record_gate "tg_ratio_long" "$R_L" "$GATE_TGRATIO_LONG" "$P" "$KIND_L" "$NOTE_L"
  else
    record_skip "tg_ratio_long" "missing TG_TQ_L or TG_Q8_L"
  fi
fi

# ----- pretty-print summary table to stderr ---------------------------------
print_gate_summary() {
  printf "\n[gate] ──────────── gate summary ────────────\n" >&2
  printf "[gate] %-18s  %-10s  %-10s  %-6s  %-4s  %s\n" \
    "gate" "value" "threshold" "result" "kind" "note" >&2
  for row in "${GATE_RESULTS[@]}"; do
    IFS='|' read -r name value threshold pass kind note <<< "$row"
    if [[ "$kind" == "skip" ]]; then
      r="SKIP"
    elif [[ "$pass" == "1" ]]; then
      r="PASS"
    else
      r="FAIL"
    fi
    printf "[gate] %-18s  %-10s  %-10s  %-6s  %-4s  %s\n" \
      "$name" "${value:--}" "${threshold:--}" "$r" "$kind" "$note" >&2
  done
  printf "[gate] HARD_FAILED=%d  SOFT_FAILED=%d\n" "$HARD_FAILED" "$SOFT_FAILED" >&2
  printf "[gate] ────────────────────────────────────────\n\n" >&2
}
print_gate_summary

# ----- summary.json ---------------------------------------------------------
# Compute acc Δ for json (best-effort, blank if either side missing).
ACC_DELTA_JSON=""
if [[ -n "$ACC_F16" && -n "$ACC_TQ" && "$ACC_F16" != "NA" && "$ACC_TQ" != "NA" ]]; then
  ACC_DELTA_JSON=$(awk -v a="$ACC_TQ" -v b="$ACC_F16" 'BEGIN{printf "%.4f", a-b}')
fi
{
  printf '{\n'
  printf '  "build_dir": "%s",\n' "$BUILD_DIR"
  printf '  "model": "%s",\n' "$MODEL"
  printf '  "wiki":  "%s",\n' "$WIKI"
  printf '  "tq_mode": "%s",\n' "${TQ_MODE:-}"
  printf '  "skip_ppl": %s, "skip_bench": %s, "skip_acc": %s,\n' \
    "$([[ $SKIP_PPL -ne 0 ]] && echo true || echo false)" \
    "$([[ $SKIP_BENCH -ne 0 ]] && echo true || echo false)" \
    "$([[ $SKIP_ACC -ne 0 ]] && echo true || echo false)"
  printf '  "ppl": { "f16": "%s", "q8_0": "%s", "tq": "%s" },\n' \
    "${PPL_F16:-}" "${PPL_Q8:-}" "${PPL_TQ:-}"
  printf '  "acc": { "source": "%s", "hellaswag_tasks": %s, "hs_ctx": %s, "hellaswag_file": "%s", "acc_json": "%s", "f16": "%s", "tq": "%s", "delta_pp": "%s" },\n' \
    "${ACC_SOURCE:-skipped}" "$HELLASWAG_TASKS" "$HS_CTX" \
    "${HELLASWAG_FILE:-}" "${ACC_JSON:-}" \
    "${ACC_F16:-}" "${ACC_TQ:-}" "${ACC_DELTA_JSON:-}"
  printf '  "tg":  { "ctx_short": %s, "ctx_long": %s, "f16_short": "%s", "q8_0_short": "%s", "tq_short": "%s", "f16_long": "%s", "q8_0_long": "%s", "tq_long": "%s" },\n' \
    "$CTX_SHORT" "$CTX_LONG" "${TG_F16_S:-}" "${TG_Q8_S:-}" "${TG_TQ_S:-}" "${TG_F16_L:-}" "${TG_Q8_L:-}" "${TG_TQ_L:-}"
  printf '  "thresholds": { "dppl_f16": %s, "dppl_f16_sanity": %s, "dppl_q8": %s, "dacc_pp": %s, "tg_ratio_long": %s, "tg_ratio_short": %s },\n' \
    "$GATE_DPPL_F16" "$GATE_DPPL_F16_SANITY" "$GATE_DPPL_Q8" "$GATE_DACC_PP" "$GATE_TGRATIO_LONG" "$GATE_TGRATIO_SHORT"
  printf '  "strict": { "ppl": %s, "q8": %s, "long": %s },\n' \
    "$([[ $STRICT_PPL -ne 0 ]] && echo true || echo false)" \
    "$([[ $STRICT_Q8 -ne 0 ]] && echo true || echo false)" \
    "$([[ $STRICT_LONG -ne 0 ]] && echo true || echo false)"
  printf '  "gate_results": [\n'
  _last=$((${#GATE_RESULTS[@]} - 1))
  _i=0
  for row in "${GATE_RESULTS[@]}"; do
    IFS='|' read -r name value threshold pass kind note <<< "$row"
    pass_json="false"; [[ "$pass" == "1" ]] && pass_json="true"
    [[ "$kind" == "skip" ]] && pass_json="null"
    note_esc="${note//\"/\\\"}"
    val_json="null"; [[ -n "$value" ]] && val_json="\"$value\""
    thr_json="null"; [[ -n "$threshold" ]] && thr_json="\"$threshold\""
    printf '    { "name": "%s", "value": %s, "threshold": %s, "pass": %s, "kind": "%s", "note": "%s" }' \
      "$name" "$val_json" "$thr_json" "$pass_json" "$kind" "$note_esc"
    [[ $_i -lt $_last ]] && printf ','
    printf '\n'
    _i=$((_i+1))
  done
  printf '  ],\n'
  printf '  "hard_failed": %d,\n' "$HARD_FAILED"
  printf '  "soft_failed": %d,\n' "$SOFT_FAILED"
  printf '  "failed": %d\n' "$HARD_FAILED"
  printf '}\n'
} > "$OUT_DIR/summary.json"
echo "[gate] summary: $OUT_DIR/summary.json"

if [[ $HARD_FAILED -ne 0 ]]; then
  echo "[gate] RESULT = FAIL ($HARD_FAILED hard / $SOFT_FAILED soft)"
  exit 1
fi
if [[ $SOFT_FAILED -ne 0 ]]; then
  echo "[gate] RESULT = PASS-soft ($SOFT_FAILED soft fail; see table)"
else
  echo "[gate] RESULT = PASS"
fi
