# FP16 管线（含 K/V F16）完整精度测试

## 目的

在启用 **GGML_REEX_FP16_PIPELINE** 且 **K/V 在注意力侧为 F16**（写入 KV 前在 `llama_kv_cache` 内转为 F32 再 `set_rows`）时，验证：

1. **Backend-ops**：rms_norm F16、softmax F16、mul_mat / mul_mat_id F16 等用例全部通过（`ggml_set_rows` 保持 **F32 源**，与上游一致；KV 在 `llama_kv_cache` 内 F16→F32 后再写入）。
2. **PPL**：与历史基线无退化（FP16 管线 PPL 在统计误差内与 F32/原实现一致）。

## 前置条件

- 编译时开启：`-DGGML_REEX_FP16_PIPELINE=ON -DGGML_USE_REEX=ON`
- 可选 CUDA：`-DGGML_CUDA=ON`（PPL 可用 GPU 加速）

## 一键运行（推荐）

```bash
# 仅 backend-ops（无需模型）
./tests/test-reex/run_fp16_pipeline_precision_test.sh

# 指定 build 目录
./tests/test-reex/run_fp16_pipeline_precision_test.sh build_fp16

# 完整测试：backend-ops + PPL（需模型与数据集）
./tests/test-reex/run_fp16_pipeline_precision_test.sh build_fp16 \
  /path/to/Qwen3-30B-A3B-Q4_K.gguf \
  /path/to/wikitext-103-raw-v1/validation.txt
```

脚本会：

1. 检查或配置 CMake（GGML_REEX_FP16_PIPELINE=ON）。
2. 编译 `test-backend-ops` 与 `llama-perplexity`。
3. 运行 **test-backend-ops**（FP16 管线相关 op；不含「set_rows 的 F16 源」，该路径已收敛到 llama 边界 cast）。
4. 若传入模型与数据，跑三种 KV 配置的 **PPL**（KV F16、K8V4、Q8Q8）并写入 `tests/test-reex/ppl_fp16_pipeline_results/`。

## 手动步骤

### 1. 配置与编译

```bash
mkdir -p build_fp16 && cd build_fp16
cmake .. -DGGML_REEX_FP16_PIPELINE=ON -DGGML_USE_REEX=ON -DCMAKE_BUILD_TYPE=Release
make -j test-backend-ops llama-perplexity
```

### 2. Backend-ops

```bash
./test-backend-ops
# 或
./tests/test-backend-ops
```

### 3. PPL（与历史报告一致）

- 数据集：wikitext-103-raw-v1 `validation.txt`（或 wiki.test.raw）。
- 建议：100 chunks，n_ctx=512，batch=512，ngl=99。

```bash
# KV F16
./bin/llama-perplexity -m model.gguf -f validation.txt -n 100 --ctx-size 512 --batch-size 512 -ngl 99 -t 1

# KV K8V4
./bin/llama-perplexity -m model.gguf -f validation.txt -n 100 --ctx-size 512 --batch-size 512 -ngl 99 -ctk q8_0 -ctv q4_0 -t 1

# KV Q8Q8
./bin/llama-perplexity -m model.gguf -f validation.txt -n 100 --ctx-size 512 --batch-size 512 -ngl 99 -ctk q8_0 -ctv q8_0 -t 1
```

## CPU 与 CUDA 验证说明

- **test-backend-ops**  
  - **默认**：遍历所有「非 CPU」backend（如 CUDA），用 **CPU 参考实现** 做一致性对比，即同时覆盖 **CPU（参考）** 与 **CUDA（被测）**。  
  - **CPU-only 构建**：默认会跳过 CPU，不跑 op 一致性；脚本会额外执行 `test-backend-ops -b CPU`，用 CPU 与 CPU 参考实现对比，完成 **CPU 端验证**。  
  - 若需单独验证某 backend，可手动：`./test-backend-ops -b CPU` 或 `./test-backend-ops -b CUDA`（设备名以运行时枚举为准）。
- **PPL**：使用当前构建的 backend（有 CUDA 时通常 `-ngl 99` 会跑在 GPU 上）；若需单独测 CPU PPL，可传 `-ngl 0`。

## 达标标准

- **Backend-ops**：全部用例 PASS；CPU 与 CUDA（若构建）均需通过。完整矩阵与「官网基线」见 `REEX_FULL_VALIDATION.md` / `scripts/run_reex_full_validation.sh`。
- **PPL**：FP16 管线三种 KV 的 PPL 与历史基线（如 `ppl_mr_fp16_pipeline_verification.md`）相比，差异在 ±0.02 内（统计误差约 ±0.118）。

## 参考

- `ppl_mr_fp16_pipeline_verification.md`：历史 FP16 管线 PPL 基线（当时 K/V 仍为 F32）。
- 技术方案文档：FP16 主数据流与 set_rows F16 支持说明。
- **KV cache（Q4Q4、K8V4 等）与 CUDA 评测**：复用 **lm_evaluator**，不在此处重复。参见 `models_config_cuda_w4int16_kv_matrix.json`、`scripts/run_full_evaluation.sh`、`scripts/run_kv_q4_unified_ppl_100chunks.sh` 及 `model_evaluation_results/cuda_w4int16/` 下的结果与基准对比。
