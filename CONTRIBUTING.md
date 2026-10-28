# 参与贡献

本文档说明 rx-met 的公开开发、验证和提交要求。完整运行环境以固定 runtime 与匹配的应用包为准；
本地开发构建和测试本次修改涉及的模块。

## 开发环境

- Python 支持范围为 3.10 至 3.12，精确定义见 `pyproject.toml`。
- 原生 AIMET、QuantGRU 和 QuantLSTM 构建要求 Linux x86_64、C++17、CMake、Ninja 及对应的
  ONNX Runtime 或 CUDA 工具链。
- 完整 GPU 验证要求 NVIDIA Driver、NVIDIA Container Toolkit 和与目标变体兼容的
  GPU。
- 依赖配置和文档修改可以在没有 GPU 的环境中检查。

项目使用 PEP 517 构建入口。wheel 构建命令为 `python3 -m pip wheel .`，完整发布
构建通过发布脚本执行。

## 修改范围

- 每个提交保持单一职责，包含同一目标的代码、测试和文档。
- 公共 Python 包继续位于 `src/`，原生 AIMET 位于 `native/`，替换算子位于
  `operators/`。
- 模型、数据集、校准样本、运行输出、动态库和本地调研文档保留在工作区或外部存储。
- 公开提交使用脱敏路径和日志；第三方代码及数据包含明确的授权和许可证信息。

## 依赖变更

Python 兼容范围和依赖声明位于 `pyproject.toml`，发布变体位于
`docker/variants.json`。修改其中任一文件后，应重新生成并检查派生产物：

```bash
python3 scripts/dependencies.py lock
python3 scripts/dependencies.py check
python3 -m tests release -k dependencies -v
```

`scripts/dependencies.py lock` 是 `docker/docker-bake.hcl` 和
`docker/requirements/*.lock` 的唯一维护入口。

## 构建和测试

所有修改至少执行：

```bash
git diff --check
python3 -m tests common release
```

根据修改范围补充以下验证：

| 修改范围 | 验证入口 |
| --- | --- |
| AIMET 原生运行库 | `docs/AIMET_ONNX_NATIVE.md` 中的构建和冒烟测试 |
| QuantGRU | `operators/quant-gru/README.md` 和 `operators/quant-gru/script/run_all_test.sh` |
| QuantLSTM 集成 | `python3 -m tests recurrent -v`（CUDA 环境） |
| 依赖矩阵或锁文件 | `python3 scripts/dependencies.py check` |
| Docker 构建或发布脚本 | `./scripts/release/build.sh <变体>` |
| 应用包 GPU 行为 | `./scripts/release/verify_bundle.sh --gpu --gpu-device 0 <变体>` |
| KWS 或 ONNX PTQ 示例 | `scripts/release/verify_*_example.sh` 对应的真实数据验证 |

提交说明记录受 GPU、模型或数据集条件限制的测试项及原因。

## 文档和版本记录

- 用户可见行为变化时，更新 `README.md` 或 `docs/` 中对应的用户文档。
- 配置字段变化时，更新 `docs/Quant_config.md` 和示例配置说明。
- 构建、依赖或发布流程变化时，更新 `docs/Release_packaging.md`。
- 非显然的架构决策使用 `docs/adr/` 记录背景、决策、影响和备选方案。
- 面向下一个版本的重要变化写入根目录 `CHANGELOG.md` 的 `[Unreleased]`（不存在时新建）。QuantGRU、QuantLSTM 的集成变更也统一记录在这里，算子目录不维护独立版本日志。
- 发布前将已纳入本次产物的 Unreleased 条目并入 `VERSION` 对应的版本，按最终行为合并重复或已被替代的记录；发布脚本从根目录 CHANGELOG 提取当前版本内容，应用包不包含其他版本或 Unreleased。

文档中的命令、路径和版本以当前提交中的文件为依据。

## 公共环境与应用包

CPU/GPU 环境使用 rx-quant-env `202610`，环境定义与脚本由本仓库维护，通过环境版本标签复用兼容镜像。
环境版本是配置中的固定 `YYYYMM` 字符串，重复或跨月构建、产品更新不自动改版。
构建来源摘要和镜像 ID 用于追踪，兼容性由环境元数据和实际依赖验证。
ORT 离线构建材料保存在 `docker/environment/third_party/`；业务原生源码按产品职责维护。
基础镜像、依赖、ABI 或工具链定义变化时，将环境版本更新为调整当月；目录和脚本调整不使已有环境失效。
发布相关修改需验证安装器、环境类型检查、归档完整性和当前版本日志，并执行真实离线安装与只读挂载验收。

## 提交说明

提交或 Pull Request 应包含：

- 变更目的和行为影响。
- 兼容性、依赖或发布制品变化。
- 已执行的测试及结果。
- 未执行的验证和剩余风险。
- 对应的文档和 Changelog 更新。

问题报告应包含 rx-met 版本、CUDA 变体、GPU 和驱动版本、最小复现步骤及完整错误
信息。提交的日志和复现材料完成敏感信息脱敏，并使用公开地址和示例路径。
