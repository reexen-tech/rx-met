# REEX 核心功能上库前放行清单

## 1. 适用范围

本文档只服务于当前这批**可认定的核心功能**上库前放行，不等同于完整 `full gate`。

当前核心功能范围：

- `Qwen3 MoE` 主链路 `FP16 pipeline`
- `KV cache` 写回边界：`F16 -> F32 -> ggml_set_rows`
- CPU/CUDA 为 `FP16 pipeline` 提供的配套激活处理
- 与上述核心链路直接相关的验证脚本与文档同步

不在本文默认放行范围内：

- `Q8 E2E` 深度专项
- `single-block/full-layer` dump 对比
- `Qwen3/Qwen3.5` profiling / 研究脚本
- 仅用于长期回归的 `full gate`

## 2. 放行原则

上库前需要同时满足两件事：

1. **官网源码路径 OK**：不开 REEX 宏时，构建与基础运行不被破坏。
2. **自研核心路径 OK**：`FP16 pipeline + KV 边界 cast + CUDA/CPU 配套` 的核心行为通过验证。

如果只满足其一，不建议放行。

## 3. 必跑项

### A. 上游基线

命令：

```bash
./scripts/reex_validate.sh baseline
```

目的：

- 覆盖 `Tier0 + Tier1`
- 确认未开 REEX 时的上游路径正常
- 确认基础自研构建矩阵未坏

通过标准：

- 脚本退出码为 `0`
- `Tier 0` 的 `test-backend-ops` 通过
- `Tier 1` 关键变体至少通过 `cpu_reex_fp16`
- 有 CUDA 环境时，关键变体至少通过 `cuda_reex_fp16`

### B. FP16 核心专项

命令：

```bash
./scripts/reex_validate.sh fp16 build_fp16 /path/model.gguf /path/wiki.txt
```

如果暂时没有模型和数据，至少先跑：

```bash
./scripts/reex_validate.sh fp16 build_fp16
```

目的：

- 验证 `FP16 pipeline`
- 验证 `KV` 边界 cast 后的行为
- 验证 `backend-ops` 与 `PPL`

通过标准：

- `test-backend-ops` 全部 PASS
- 有 CUDA 时，CUDA backend 不出现真实功能性失败
- 若跑 PPL：`KV F16`、`K8V4`、`Q8Q8` 与历史基线相比 `|ΔPPL| <= 0.02`

### C. CUDA 核心专项

命令：

```bash
./scripts/reex_validate.sh cuda-ab
```

目的：

- 验证 CUDA A/B 核心路径
- 排除“编译过了，但实际走错路径/回退路径/性能异常”的情况

通过标准：

- 脚本退出码为 `0`
- 关键 CUDA 变体至少覆盖：
  - `cuda_only`
  - `cuda_reex_fp16`
  - `cuda_reex_gemm`
  - `cuda_reex_full`
- 若脚本包含自动降 `ngl` / 调整 batch 的自愈逻辑，最终结果仍需为通过，而不是仅仅“重试过”

## 4. 条件性必跑项

### 改到 GEMM Q8 / KV Q8-Q4 / FA 全量量化模板

额外命令：

```bash
VERIFY_GEMM_Q8=1 ./scripts/reex_validate.sh baseline
./scripts/reex_validate.sh q8-e2e
```

### 改到验证脚本本身

额外要求：

- 至少重新跑一遍 `baseline`
- 若改到 `fp16` / `cuda-ab` / `gate-all` 入口，同时重跑对应入口

## 5. 当前建议放行组合

对于这批核心功能，建议按下面组合放行：

```bash
./scripts/reex_validate.sh baseline
./scripts/reex_validate.sh fp16 build_fp16 /path/model.gguf /path/wiki.txt
./scripts/reex_validate.sh cuda-ab
```

若本次合并同时包含脚本重构、门禁入口改造或高风险 CUDA 调整，再补：

```bash
./scripts/reex_validate.sh gate-project build_fp16 /path/q4_model.gguf /path/wiki.txt
```

说明：

- 这里默认用 `gate-project` 作为“项目可执行的快速上库门禁”
- `full gate` 保留给夜间回归、发布前兜底或专项确认

## 6. 明确不作为放行证据的情况

以下情况不能视为“已充分验证”：

- 只做了 `git push` / `MR` / 文档整理，没有对冻结代码快照做实际跑测
- 验证过程中脚本本身还在持续修改，但没有在修完后对最终版本重跑
- 只有 `help` / `--list-ops` / 编译通过，没有 `backend-ops + PPL + CUDA A/B`
- 关键阶段被 `skip`，且不是本文定义的“非默认范围”
- 只看单次对话记录，没有留存最终测试结果

## 7. 结果记录模板

每次准备上库时，建议把结果按下面模板留档：

```md
# REEX 核心功能放行记录

- 代码分支：
- 目标提交：
- 测试日期：
- 执行人：
- 环境：
  - 宿主机 / Docker：
  - CUDA 版本：
  - 模型：
  - 数据：

## 执行项

- [ ] `baseline`
- [ ] `fp16 build_fp16`
- [ ] `fp16 build_fp16 + PPL`
- [ ] `cuda-ab`
- [ ] `q8-e2e`（如适用）

## 结果

- `baseline`：
- `fp16 backend-ops`：
- `fp16 PPL`：
- `cuda-ab`：
- `q8-e2e`：

## 关键结论

- 官网源码路径是否 OK：
- 自研核心路径是否 OK：
- 是否允许上库：

## 备注

- 已知问题：
- 本次跳过项及原因：
```

## 8. 最终放行判断

只有当下面两项都成立时，才建议点头上库：

- `官网源码路径 OK`
- `自研核心路径 OK`

推荐最终结论只写两种之一：

- `允许上库`
- `不允许上库，需补测/修复后再审`
