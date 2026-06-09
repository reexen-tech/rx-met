# REEX / 自研路径 — 完整验证规范

## 零、运行环境（nvcc / PATH）

| 场景 | 做法 |
|------|------|
| 宿主机已装 CUDA | 确保 `nvcc -V` 可用；若否，执行 `export PATH=/usr/local/cuda/bin:$PATH`（或你的 `cuda-*` 目录），必要时 `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`。`verify_build_variants.sh` / `run_reex_full_validation.sh` 会尝试自动追加 `/usr/local/cuda/bin`。 |
| 宿主机**无** nvcc（如仅 SSH CI、精简 PATH） | 使用已有 **CUDA 镜像容器**（推荐与评测一致）：`./scripts/run_reex_full_validation_docker.sh`。默认容器名 `llq-eval`，工作目录 `/workspace/llama.cpp`。需 `docker start llq-eval` 且挂载本仓库。 |
| 强制仅 CPU | `SKIP_CUDA=1 ./scripts/run_reex_full_validation.sh` |

**注意**：Cursor/部分自动化环境的 `PATH` 可能极短，若未进容器又未设 `SKIP_CUDA=1`，会因找不到 `nvcc` 导致 CUDA 变体失败——此时请用 Docker 入口或补齐 PATH。

---

本文档供**合并代码后**或**同事修改 REEX 相关逻辑**时执行，用于确认：

1. **官网源码路径**：不启用自研宏时，行为与上游一致，可编译、可跑基础测试。  
2. **自研全路径**：启用 `GGML_USE_REEX`、`GGML_REEX_FP16_PIPELINE`、可选 CUDA/GEMM 等组合时，编译与测试通过。

---

## 一、CMake 选项与编译期宏（必读）

| CMake 选项 | 默认 | 主要编译宏 | 含义（简述） |
|------------|------|------------|--------------|
| *(无)* | — | — | **上游基线**：不启用 REEX。 |
| `GGML_USE_REEX` | OFF | `GGML_USE_REEX` | PWNL/LUT：sin/cos/silu 等分段近似。 |
| `GGML_REEX_FP16_PIPELINE` | OFF | `GGML_REEX_FP16_PIPELINE` | 主数据流 F16 边界（Qwen3 MoE 等 + CPU/CUDA 若干 op 的 F16 路径）；**KV 写入在 `llama_kv_cache` 内 `F16→F32` 后再走上游 `ggml_set_rows`**。 |
| `GGML_REEX_GEMM` | OFF | `GGML_USE_REEX_GEMM` 等 | W4×Q8/Q16/FLOAT GEMM 后端；**与 `GGML_CPU_REPACK=ON` 互斥**。 |
| `GGML_REEX_GEMM_ACTIVATION` | Q16 | `GGML_REEX_GEMM_ACTIVATION_Q8` / `_Q16` / `_FLOAT` | GEMM 激活位宽。 |
| `GGML_CUDA` | OFF | `GGML_USE_CUDA` | CUDA 后端。 |
| `GGML_REEX_KV_EXPERIMENTAL` | OFF | `GGML_REEX_KV_EXPERIMENTAL` | KV 实验特性（一般验证矩阵可不测，除非你在改相关代码）。 |

**关系说明**

- **`GGML_REEX_FP16_PIPELINE` 常与 `GGML_USE_REEX=ON` 一起开**：PWNL + F16 主线是当前自研推理的默认组合。  
- **宏只影响编译进分支的代码**；同一源码树可同时保留多条 build 目录，互不干扰。

---

## 二、验证层级（按顺序执行）

在当前仓库里，REEX 相关“功能测试”**完整集合**包括：

1. **Tier 0 上游基线**：无 REEX 宏时的 `test-backend-ops` + `llama-perplexity -h`
2. **Tier 1 自研矩阵**：多 CMake 组合的编译 + 最小运行时
3. **Tier 2 FP16 pipeline 专项**：`test-backend-ops` + 可选 PPL
4. **REEX LUT 精度专项**：`test-reex-lut`
5. **REEX GEMM CPU 专项**：`test-reex-gemm`
6. **各变体 Wikitext PPL 冒烟**：`verify_variants_wikitext_smoke.sh`
7. **CUDA 专项**：`verify_build_cuda_reex.sh`
8. **GEMM Q8 端到端专项**：`verify_reex_gemm_q8_e2e.sh`
9. **full-layer / single-block dump 对比**：`run_single_block_compare.sh`

如果目标是“同事改完代码后，一次性跑完当前所有不同功能测试”，上面这些不应遗漏。

### Tier 0 — 上游源码路径（必须通过）

**目的**：确认未开 REEX 时，与 ggml-org 期望一致，避免「只有自研开关能编过」。

| 步骤 | CPU 基线 | CUDA 基线（有 GPU/nvcc 时） |
|------|----------|-----------------------------|
| 配置 | `cmake -S . -B build_upstream_cpu -DCMAKE_BUILD_TYPE=Release` | 同上 + `-DGGML_CUDA=ON`（目录可名 `build_upstream_cuda`） |
| 编译 | `cmake --build build_upstream_cpu -j$(nproc) --target llama-perplexity test-backend-ops` | 同上 |
| 测试 | `build_upstream_cpu/bin/test-backend-ops`（或 `build_upstream_cpu/tests/test-backend-ops`，以实际产物为准） | 同上 |
| 冒烟 | `build_upstream_cpu/bin/llama-perplexity -h` 有正常输出 | 同上，必要时设置 `LD_LIBRARY_PATH` 含 CUDA `lib64` |

**通过准则**：`test-backend-ops` **全部 PASS**；`llama-perplexity` 能拉起帮助信息。

**说明**：Tier 0 **不要**传 `-DGGML_USE_REEX=ON` / `-DGGML_REEX_FP16_PIPELINE=ON`。

---

### Tier 1 — 自研矩阵：编译 + 最小运行时（必须通过）

**目的**：覆盖常见 REEX 组合，与 CI/本地夜测对齐。

**推荐一键**（仓库根目录）：

```bash
./scripts/reex_validate.sh baseline
# 或宿主机无 nvcc 时跑完整 Tier0 + Tier1：
./scripts/reex_validate.sh baseline-docker
# 或直接：
./scripts/run_reex_full_validation.sh
# 或宿主机无 nvcc 时：
./scripts/reex_validate.sh matrix-docker
```

或手动调用多变体脚本（与上脚本 Tier 1 部分等价）：

```bash
# 全矩阵（含多种 CPU/CUDA REEX 组合，耗时较长）
./scripts/verify_build_variants.sh

# 快速子集：cpu_default、cpu_reex_fp16、cuda_only、cuda_reex_fp16
VERIFY_QUICK=1 ./scripts/verify_build_variants.sh

# 无 GPU
SKIP_CUDA=1 ./scripts/verify_build_variants.sh

# 仅编译、不跑二进制冒烟
VERIFY_SKIP_RUNTIME=1 ./scripts/verify_build_variants.sh
```

**通过准则**：脚本退出码 **0**；日志中「已通过变体」含至少 `cpu_reex_fp16`；有 CUDA 时还应含 `cuda_reex_fp16`。

**可选加强**：`VERIFY_GEMM_Q8=1` 可对 Q8 激活多两个变体（见 `verify_build_variants.sh` 头部注释）。

---

### Tier 2 — FP16 管线：backend-ops 全量 + 可选 PPL（修改了 FP16/KV/算子 时强烈建议）

**目的**：在 **开启 `GGML_REEX_FP16_PIPELINE`** 的构建上跑 **完整 `test-backend-ops`**（含 RMSNorm/Softmax/MulMat 等 F16 用例，**不再依赖「set_rows 的 F16 源」**——与上游一致，KV 边界在 llama）。  

```bash
# 仅 backend-ops（无需模型）
./tests/test-reex/run_fp16_pipeline_precision_test.sh build_fp16

# 加上 PPL（需 GGUF + wikitext 等）
./tests/test-reex/run_fp16_pipeline_precision_test.sh build_fp16 \
  /path/to/model-Q4_K.gguf /path/to/wikitext-103-validation.txt
```

**通过准则**：

- `test-backend-ops` 全部 PASS；  
- 若跑 PPL：与 `ppl_mr_fp16_pipeline_verification.md` 中基线相比，|ΔPPL| 建议在 **0.02** 内（统计误差约 ±0.12 量级时以多次运行为准）。

---

### Tier 3 — 可选：Wikitext 冒烟 / CUDA 轨迹 / GEMM E2E

在 Tier 1 构建产物就绪后，按需使用（详见各脚本注释）：

| 脚本 | 用途 |
|------|------|
| `VERIFY_WIKITEXT_SMOKE=1 VERIFY_MODEL=... VERIFY_WIKITEXT=... ./scripts/verify_build_variants.sh` | 各变体编译通过后，小批量 PPL（见 `verify_variants_wikitext_smoke.sh`） |
| `./scripts/verify_build_cuda_reex.sh` | CUDA REEX A/B、batch、`test-reex-cuda-q16`、`eval_single_token` trace sweep |
| `./scripts/verify_reex_gemm_q8_e2e.sh` | GEMM Q8 端到端对比 |

---

## 三、环境变量速查

| 变量 | 说明 |
|------|------|
| `VERIFY_ROOT` | 仓库根（默认脚本父目录） |
| `VERIFY_JOBS` | 并行编译线程数 |
| `SKIP_CUDA=1` | 跳过所有 CUDA 变体 |
| `VERIFY_QUICK=1` | 仅跑缩小矩阵 |
| `VERIFY_SKIP_RUNTIME=1` | 编译后不跑 `llama-perplexity -h` / `test-backend-ops --list-ops` |
| `VERIFY_GEMM_Q8=1` | 增加 Q8 激活变体 |
| `N_CHUNKS` | FP16 PPL 脚本 chunk 数（默认 100） |

---

## 四、推荐给 Code Review 的检查清单

- [ ] **Tier 0**：无 REEX 构建 + `test-backend-ops` 通过  
- [ ] **Tier 1**：`./scripts/run_reex_full_validation.sh` 或 `verify_build_variants.sh` 通过  
- [ ] 若改动 **FP16 / KV / ggml op**：**Tier 2** `run_fp16_pipeline_precision_test.sh` 至少跑通 backend-ops  
- [ ] 若改动 **CUDA kernel / 调度**：加跑 `verify_build_cuda_reex.sh` 或自测 `test-reex-cuda-q16`  
- [ ] 文档与注释中若提到「set_rows F16 源」，应已与当前实现一致：**KV 为 F16→`ggml_cast`→F32→`set_rows`**

---

## 五、相关文档与入口

| 文件 | 内容 |
|------|------|
| `scripts/reex_validate.sh` | **统一入口**：把 baseline / matrix / fp16 / cuda-ab / q8-e2e 收敛成少量子命令 |
| `scripts/reex_validate.sh gate-project` | **默认项目门禁**：适合日常开发与快速上库验证 |
| `scripts/reex_validate.sh gate-full` | **完整深度回归门禁**：保留全量功能测试覆盖 |
| `scripts/run_reex_all_function_tests.sh` | **全功能门禁**：把当前所有功能测试类别尽量串成一条完整流程 |
| `scripts/run_reex_all_function_tests_docker.sh` | **Docker 全功能门禁**：宿主机无 nvcc 时的一键全跑入口 |
| `scripts/REEX_VALIDATION_GUIDE.md` | **维护手册**：按改动类型选测试、建议合并前最小验证集 |
| `scripts/run_reex_full_validation.sh` | **推荐总入口**：Tier 0 + Tier 1 |
| `scripts/verify_build_variants.sh` | 多变体 CMake 矩阵 |
| `tests/test-reex/run_fp16_pipeline_precision_test.sh` | Tier 2 |
| `README_fp16_pipeline_precision.md` | FP16 管线步骤与达标标准 |
| `ppl_mr_fp16_pipeline_verification.md` | PPL 历史基线 |
