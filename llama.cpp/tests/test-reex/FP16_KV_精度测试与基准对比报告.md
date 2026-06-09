# FP16 管线（含 K/V F16）精度测试与基准对比报告

**执行时间**: 2026-03-19  
**Build**: `build_reex_ppl`，`GGML_REEX_FP16_PIPELINE=ON`，`GGML_USE_REEX=ON`  
**代码**: 自研路径 Q/K/V 均 `reex_cast_f16`；KV 写入在 `llama_kv_cache` 内 `ggml_cast(F16→F32)` 后使用上游 `ggml_set_rows(F32 源)`（ggml 核心与官网一致，便于同步）

---

## 1. Backend-ops 测试结果 ✅

| 项目 | 结果 |
|------|:----:|
| 可执行文件 | `build_reex_ppl/bin/test-backend-ops` |
| 运行命令 | `./bin/test-backend-ops test` |
| 结果 | **1/1 backends passed，OK** |
| 覆盖 | 含 SET_ROWS、RMS_NORM、SOFT_MAX、MUL_MAT、MUL_MAT_ID 等 103 个 op |

**说明**：当前为单 backend（CPU）构建，测试在 CPU 上执行全部用例（含新增的 set_rows F16 源用例），全部通过。

---

## 2. 基准对比（PPL）

### 2.1 历史基线（来自 `ppl_mr_fp16_pipeline_verification.md`）

当时为 **K/V 保持 F32** 的 FP16 管线，100 chunks，n_ctx=512，batch=512，ngl=99：

| KV 配置 | 基线 PPL | 标准差 |
|---------|----------|--------|
| KV F16  | 7.0445   | ±0.118 |
| KV K8V4  | 7.0542   | ±0.118 |
| KV Q8Q8  | 7.0375   | ±0.118 |

### 2.2 本次 PPL 运行说明

仓库内未包含模型与数据集，**未在本机自动跑 PPL**。请在本地执行以下命令完成 PPL 与基准对比：

```bash
# 使用本次编译的 build_reex_ppl
BUILD=/home/llq/workspace/llama.cpp/build_reex_ppl
MODEL=/path/to/Qwen3-30B-A3B-Q4_K.gguf
DATA=/path/to/wikitext-103-raw-v1/validation.txt

# 三种 KV 配置（与基线一致）
$BUILD/bin/llama-perplexity -m $MODEL -f $DATA -n 100 --ctx-size 512 --batch-size 512 -ngl 99 -t 1
$BUILD/bin/llama-perplexity -m $MODEL -f $DATA -n 100 --ctx-size 512 --batch-size 512 -ngl 99 -ctk q8_0 -ctv q4_0 -t 1
$BUILD/bin/llama-perplexity -m $MODEL -f $DATA -n 100 --ctx-size 512 --batch-size 512 -ngl 99 -ctk q8_0 -ctv q8_0 -t 1
```

### 2.3 达标判定（与基线对比）

- 统计误差约 **±0.118**。
- **达标**：各 KV 配置下本次 PPL 与上表基线差异 **|ΔPPL| < 0.02**（建议 < 0.01 更稳妥）。
- 若 |ΔPPL| 在 0.02～0.06 之间，可结合多次运行判断；若 > 0.06 需排查。

---

## 3. 结论

| 验证项 | 状态 |
|--------|:----:|
| Backend-ops（含 set_rows F16） | ✅ 已执行，全部通过 |
| PPL 与历史基线对比 | ⏳ 需在本地提供模型与数据后运行上述命令并对比 |

**建议**：在提供 Qwen3-30B-A3B-Q4_K.gguf 与 wikitext-103-raw-v1 validation 后，跑完三种 KV 的 PPL，将结果与 2.1 节基线对比即可完成完整精度与基准对比验证。

---

## 4. KV Cache（Q4Q4、K8V4 等）与 CUDA 评测：复用 lm_evaluator

**避免重复**：KV 均为 Q4、K Q8 + V Q4 等 CUDA 上的 PPL/评测及与基准对比，已在 **lm_evaluator** 中实现，不在此仓库重复维护脚本。

- **配置**：`lm_evaluator/models_config_cuda_w4int16_kv_matrix.json`（含 KV-f16、KV-q8_0、KV-q8_0-q4_0、KV-q4_0 等）。
- **运行**：在 lm_evaluator 目录下用 `scripts/run_full_evaluation.sh` 或指定该 config；或使用 `scripts/run_kv_q4_unified_ppl_100chunks.sh` 做 100 chunks 统一 PPL。
- **结果**：`lm_evaluator/model_evaluation_results/cuda_w4int16/` 下的 `perplexity_*_validation.json`、`model_evaluation_matrix.json` 等，可直接与 2.1 节基线或技术方案文档中的 Stage 1/1.5/3 结果对比。
