# REEX 验证使用指南

这份文档面向**后续维护者**，目标不是解释某个单独脚本，而是回答两个问题：

1. 现在这么多脚本，**平时该从哪个入口开始**？
2. 改了不同类型的代码，**最少该跑哪些测试**，才能较稳地避免把项目改坏？

## 推荐原则

- **日常只记一个入口**：`./scripts/reex_validate.sh`
- **底层脚本保留**：`run_reex_full_validation.sh`、`verify_build_variants.sh`、`verify_build_cuda_reex.sh`、`run_fp16_pipeline_precision_test.sh` 等继续存在，便于专项调试
- **按改动类型选测试**：不是每次都把所有长流程全跑一遍，而是先跑必需集，再按风险加测

## 当前“功能测试”全量清单

下面这些，才算当前 REEX 相关的**完整功能测试集合**：

| 类别 | 入口 | 覆盖点 |
|------|------|--------|
| 上游基线 | `run_reex_full_validation.sh` 的 Tier0 | 无 REEX 宏时，官网路径编译/运行是否正常 |
| 自研矩阵 | `verify_build_variants.sh` | CPU/CUDA + LUT/FP16/GEMM/Q8 等 CMake 组合是否都能编过并最小运行 |
| FP16 专项 | `run_fp16_pipeline_precision_test.sh` | FP16 pipeline 的 backend-ops + 可选 PPL |
| LUT 专项 | `test-reex-lut` | `GGML_USE_REEX` 下 LUT 精度 |
| GEMM CPU 专项 | `test-reex-gemm` | W4×FLOAT/Q16/Q8 的 CPU 精度与链路正确性 |
| Wikitext 冒烟 | `verify_variants_wikitext_smoke.sh` | 各 build_verify_* 的小批量 PPL |
| CUDA 专项 | `verify_build_cuda_reex.sh` | CUDA A/B、batch、`test-reex-cuda-q16`、`eval_single_token` trace |
| Q8 E2E 专项 | `verify_reex_gemm_q8_e2e.sh` | `test-reex-cuda-q8` + Q8/K8V4/KVQ4 的端到端性能与 PPL |
| Full-layer 对比 | `run_single_block_compare.sh` | gold / W4×float / W4×Q16 / W4×Q8 / LUT 的全层 dump 对比 |

也就是说，**如果要说“一次性把当前所有不同功能测试都跑完”**，上面这些都不能漏。

## 脚本分层

| 层级 | 推荐入口 | 目的 | 适用时机 |
|------|-----------|------|----------|
| 统一入口 | `./scripts/reex_validate.sh` | 给开发者一个稳定命令面 | 日常使用 |
| Tier0+1 | `./scripts/run_reex_full_validation.sh` | 上游基线 + 自研矩阵 | 合并前、较大改动后 |
| Tier1 | `./scripts/verify_build_variants.sh` | 多变体编译 + 最小运行时冒烟 | 快速确认构建矩阵没坏 |
| Tier2 | `./tests/test-reex/run_fp16_pipeline_precision_test.sh` | FP16 pipeline backend-ops + 可选 PPL | 改了 FP16/KV/算子 |
| Tier3 CUDA | `./scripts/verify_build_cuda_reex.sh` | CUDA A/B、batch、trace | 改了 CUDA kernel/调度 |
| Tier3 Q8 | `./scripts/verify_reex_gemm_q8_e2e.sh` | GEMM Q8 / KV Q8-Q4 端到端 | 改了 GEMM Q8 或量化组合 |
| 项目门禁 | `./scripts/reex_validate.sh gate-project` | 默认快速门禁，面向日常开发与快速上库验证 | 日常修改 / 合并前自测 |
| 项目门禁（Docker） | `./scripts/reex_validate.sh gate-project-docker` | 宿主机无 nvcc 时的默认快速门禁 | 容器化开发/远端机 |
| 全功能门禁 | `./scripts/reex_validate.sh gate-full` | 把当前所有功能测试类别串起来 | 发布前 / 大改动 / 需要一次性兜底 |
| 全功能门禁（Docker） | `./scripts/reex_validate.sh gate-full-docker` | 宿主机无 nvcc 时的 full gate | 容器化开发/远端机 |

## 最常用命令

```bash
# 1) 最推荐：上游基线 + 自研矩阵
./scripts/reex_validate.sh baseline

# 2) 宿主机没有 nvcc，用容器跑完整 Tier0 + Tier1
./scripts/reex_validate.sh baseline-docker

# 3) 只想跑自研矩阵，不跑上游 Tier0
./scripts/reex_validate.sh matrix

# 4) 宿主机没有 nvcc，用容器跑矩阵
./scripts/reex_validate.sh matrix-docker

# 5) 改了 FP16/KV/算子，跑 FP16 专项
./scripts/reex_validate.sh fp16 build_fp16

# 6) 再加模型和数据，顺带跑 PPL
./scripts/reex_validate.sh fp16 build_fp16 /path/model.gguf /path/wiki.txt

# 7) 改了 CUDA kernel / 调度
./scripts/reex_validate.sh cuda-ab

# 8) 改了 GEMM Q8 / KV Q8-Q4
./scripts/reex_validate.sh q8-e2e

# 9) 默认快速门禁（推荐日常使用）
./scripts/reex_validate.sh gate-project build_fp16 /path/q4_model.gguf /path/wiki.txt

# 10) 一次性跑“当前所有功能测试”
./scripts/reex_validate.sh gate-full build_fp16 /path/q4_model.gguf /path/wiki.txt
# 若还要补 full-layer 对比，再额外给：
MODEL_F16=/path/f16_model.gguf MODEL_Q4_K=/path/q4_model.gguf \
./scripts/reex_validate.sh gate-full build_fp16 /path/q4_model.gguf /path/wiki.txt

# 11) 宿主机无 nvcc，直接在容器里跑默认快速门禁
./scripts/reex_validate.sh gate-project-docker build_fp16 /workspace/llama.cpp/models/.../q4_k.gguf /workspace/llama.cpp/datasets/.../wiki.txt

# 12) 宿主机无 nvcc，直接在容器里一键全跑
MODEL_F16=/workspace/llama.cpp/models/.../f16.gguf \
MODEL_Q4_K=/workspace/llama.cpp/models/.../q4_k.gguf \
./scripts/reex_validate.sh gate-full-docker build_fp16 /workspace/llama.cpp/models/.../q4_k.gguf /workspace/llama.cpp/datasets/.../wiki.txt
```

## 按改动类型选测试

### 1. 只改文档、注释、验证脚本

最少运行：

```bash
./scripts/reex_validate.sh baseline
```

目的：

- 确认脚本/文档修改没有破坏默认验证入口

### 2. 改 `src/` 层图构建、模型接线、KV 边界

典型文件：

- `src/models/qwen3moe.cpp`
- `src/llama-kv-cache.cpp`
- `src/llama-graph.cpp`

最少运行：

```bash
./scripts/reex_validate.sh baseline
./scripts/reex_validate.sh fp16 build_fp16
```

原因：

- 这类改动容易让**上游路径**和**自研 FP16 路径**分叉
- 仅跑 build matrix 不足以覆盖行为变化

### 3. 改 FP16 pipeline、CPU op、KV/FP16 相关行为

典型文件：

- `ggml/src/ggml-cpu/ops.cpp`
- `ggml/src/ggml-cpu/repack.cpp`
- `ggml/src/ggml-cpu/ggml-cpu.c`
- `src/models/qwen3moe.cpp`
- `src/llama-kv-cache.cpp`

最少运行：

```bash
./scripts/reex_validate.sh baseline
./scripts/reex_validate.sh fp16 build_fp16 /path/model.gguf /path/wiki.txt
```

原因：

- `test-backend-ops` 能发现算子级不一致
- PPL 能发现图语义没崩但数值慢性退化的问题

### 4. 改 CUDA kernel、调度、Flash-Attn、supports_op

典型文件：

- `ggml/src/ggml-cuda/*.cu`
- `ggml/src/ggml-cuda/ggml-cuda.cu`

最少运行：

```bash
./scripts/reex_validate.sh baseline
./scripts/reex_validate.sh fp16 build_fp16 /path/model.gguf /path/wiki.txt
./scripts/reex_validate.sh cuda-ab
```

原因：

- 这类改动很容易出现“编译过、help 能跑、但真实推理慢/错/回退到 CPU”的情况
- `verify_build_cuda_reex.sh` 会补上 A/B、batch、trace 视角

### 5. 改 GEMM Q8、KV Q8/Q4 组合、FA 全量量化模板

典型文件：

- `ggml/src/ggml-cuda/reex/reex_gemm_cuda.cu`
- `scripts/verify_reex_gemm_q8_e2e.sh` 相关逻辑
- `CMakeLists.txt` 中 `GGML_CUDA_FA_ALL_QUANTS`

最少运行：

```bash
VERIFY_GEMM_Q8=1 ./scripts/reex_validate.sh baseline
./scripts/reex_validate.sh q8-e2e
```

原因：

- 这类改动的风险主要在**真实量化组合**和 **Flash-Attn 模板覆盖**
- 仅靠 `cuda_reex_fp16` 冒烟不够

## 建议的合并前标准

### 小改动

- `baseline`

### 中等改动

- `baseline`
- `fp16`

### 高风险改动

- `baseline`
- `fp16 + PPL`
- `cuda-ab`
- 如涉及 Q8，再跑 `q8-e2e`

### 发布前 / 想一次性兜底

- `gate-full`
- 若需要全层 dump 对比，则再提供 `MODEL_F16` + `MODEL_Q4_K`

## 环境建议

### 宿主机有 CUDA

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
```

### 宿主机无 nvcc

优先用容器：

```bash
docker start llq-eval
./scripts/reex_validate.sh matrix-docker
```

更完整的环境说明见：

- `tests/test-reex/REEX_FULL_VALIDATION.md`

## 维护建议

- **统一入口只做分发**，不要把复杂验证逻辑都塞进 `reex_validate.sh`
- **专项逻辑继续留在原脚本**，例如 CUDA A/B、Q8 E2E、FP16 PPL
- **默认优先记 `gate-project`，不是 `gate-full`**
- `gate-full` 保留为深度回归，不应用作日常快速门禁
- **新增验证脚本时，优先补这两处**
  - `scripts/reex_validate.sh`
  - 本文档的“按改动类型选测试”

这样后续同事即使不知道所有宏和脚本细节，也能按改动类型跑到足够的验证集。
