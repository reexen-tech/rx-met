# rx-met 用户使用指南

本文档说明如何选择、启动 rx-met 镜像，并运行 PyTorch 或 ONNX 量化流程。
文中的版本和命令以当前仓库为准。

相关文档：

- 量化配置详解：[`docs/Quant_config.md`](./Quant_config.md)
- 构建与发布：[`docs/Release_packaging.md`](./Release_packaging.md)
- 项目总览：[`README.md`](../README.md)

本仓库面向小模型量化。

---

## 1. rx-met 是什么

rx-met 是面向 Linux x86_64 NVIDIA GPU 的**小模型量化工具包**，包含定制
AIMET（`aimet_torch`、`aimet_onnx`、`aimet_common`）和 QuantGRU。项目通过独立
Docker 镜像提供完整运行环境。

| 工作流 | 适用对象 | 底层引擎 | 入口 | 对应章节 |
|--------|----------|----------|------|----------|
| **普通模型量化** | CNN / RNN / KWS 等中小模型（PyTorch） | `aimet_torch` | `examples/quick_start_kws.py` | [第 5.2 节](#52-运行示例脚本) |
| **ONNX 直量化 PTQ** | 已有 ONNX、无 PyTorch 训练图 | `aimet_onnx` | `examples/onnx_ptq_quick_start.py` | [第 5.4 节](#54-已有-onnx-模型的-ptq) |

选择原则：

- 模型能在单卡显存内用 PyTorch 训练 / 推理，需要 **QAT、混合精度、Power-of-2 量化、ONNX 导出** → 走「普通模型」流程。
- 已有 ONNX 模型需要 **PTQ + 编译器 encodings** → 走「ONNX 直量化 PTQ」。

---

## 2. 支持平台与版本要求

三个镜像都是 Linux x86_64 GPU 镜像，但 CUDA、Python、Torch 和 ONNX Runtime
版本不同：

| 镜像变体 | 系统 / Python | CUDA | PyTorch | ONNX Runtime GPU | 严格最低 NVIDIA Driver |
|------|------|------|------|------|------|
| `cu118` | Ubuntu 22.04 / 3.10 | 11.8 | 2.7.1 | 1.20.1 | 520.61.05 |
| `cu126` | Ubuntu 22.04 / 3.10 | 12.6 | 2.8.0 | 1.23.2 | 560.35.05 |
| `cu130` | Ubuntu 24.04 / 3.12 | 13.0 | 2.10.0 | 1.27.0 | 580.126.20 |

宿主机需要 Docker、NVIDIA Driver 和 NVIDIA Container Toolkit。镜像已经包含对应
版本的 CUDA 运行环境。镜像变体由宿主机驱动版本和 GPU 计算能力共同决定；宿主机
`nvcc --version` 显示本地 CUDA Toolkit 版本。`cu130` 包含 `sm_120` 目标；
实验性 ExportedProgram API 适用于 `cu126` 和 `cu130`。

`QuantGRU` 和 CUDA 版 ONNX Runtime 已安装在镜像中，可直接导入。ONNX PTQ
默认要求 `CUDAExecutionProvider`。

---

## 3. 加载并启动镜像

当前公开安装入口是发布制品中的离线镜像归档。从源码构建镜像见
[`Release_packaging.md`](./Release_packaging.md)。

以 `cu126` 为例，先校验制品并加载：

```bash
sha256sum --check --strict SHA256SUMS
zstd -dc rx-met-v1.0.0-cu126-linux-amd64.tar.zst | docker load
```

启动时挂载工作目录和数据集：

```bash
docker run --gpus all --rm -it \
  --ipc=host \
  --user "$(id -u):$(id -g)" \
  --group-add "$(stat -c '%g' /path/to/datasets)" \
  -e HOME=/tmp \
  -e USER="$(id -un)" \
  -e LOGNAME="$(id -un)" \
  -v /path/to/workspace:/workspace \
  -v /path/to/datasets:/datasets:ro \
  -w /workspace \
  rx-met:1.0.0-cu126
```

`--group-add` 让非 root 容器继承数据目录的只读组权限，适用于 CIFS/NFS
挂载存储；数据通过 `:ro` 挂载。容器身份由 `--user` 指定，`USER` 和 `LOGNAME`
用于初始化 PyTorch 缓存目录。

验证环境：

```bash
python3 -c "import torch, aimet_torch, aimet_onnx, quant_gru; print(torch.__version__, torch.version.cuda)"
```

命令退出码为 0，并打印与所选镜像匹配的 PyTorch 和 CUDA 版本，即表示基础环境可用。
发布镜像已经包含完整运行依赖。开发侧构建见
[`docs/Release_packaging.md`](./Release_packaging.md)。

---

## 4. 环境变量与路径配置

| 变量 / 路径 | 作用 | 适用流程 |
|-------------|------|----------|
| `RX_MET_SPEECH_COMMANDS_ROOT` | Speech Commands 数据集根目录，默认 `/datasets/speech_commands_v0.02` | 普通模型 |
| `RX_MET_ONNX_PTQ_ROOT` | ONNX 示例数据集根目录，默认 `/datasets` | ONNX PTQ |
| `RX_MET_ONNX_PTQ_EXAMPLE` | 示例名称，默认 `mobilenetv2` | ONNX PTQ |
| `RX_MET_ONNX_PTQ_MODEL` / `RX_MET_ONNX_PTQ_CALIB` | 覆盖模型与校准目录 | ONNX PTQ |
| `RX_MET_ONNX_PTQ_DEVICE` | `cuda`（默认）或显式设置为 `cpu` | ONNX PTQ |
| 镜像内示例目录 | `/opt/rx-met/examples` | 两者 |
| 推荐输出目录 | volume 挂载的 `/workspace` | 两者 |

---

## 5. 量化流程

本章基于 `examples/quick_start_kws.py`（SpeechCommands + QuantGRU 关键词识别），演示 AIMET v2 的完整量化流程。该脚本的 `main()` 内联了标准 AIMET API，可作为接入模板。

### 5.1 完整流程总览

```
FP 训练 → prepare_model → QuantizationSimModel + 混合精度位宽
  → compute_encodings 校准 (PTQ)
  → apply_power_of_2_workflow (NPU 友好的 Po2 量化)
  → freeze_quantizer_parameters + QAT 微调
  → 保存 state_dict + ONNX + .encodings 三件套
  → 重建 sim → load_state_dict → load_quantizer_encodings → 精度对比
```

### 5.2 运行示例脚本

```bash
cp -a /opt/rx-met/examples /workspace/rx-met-examples
cd /workspace/rx-met-examples
export RX_MET_SPEECH_COMMANDS_ROOT=/datasets/speech_commands_v0.02
python3 quick_start_kws.py
```

脚本会依次打印 8 个步骤的耗时与精度汇总（浮点 / PTQ / Po2 / QAT / 重载）。命令
退出码为 0，并输出“量化流程完成（已通过加载验证）”，即表示示例成功。生成文件位于
`RX_MET_KWS_OUTPUT_DIR`；未设置时使用 `examples/output/quick_start_kws/`。

### 5.3 配置和一致性

完整 API 示例由 `examples/quick_start_kws.py` 维护，配置字段和最小代码片段见
[`Quant_config.md`](./Quant_config.md)。用户指南说明两类配置的职责边界：

普通模型采用**双层配置**，详细字段说明见 [`Quant_config.md`](./Quant_config.md)：

| 配置 | 传入时机 | 控制内容 | 命名规则 |
|------|----------|----------|----------|
| **基础配置（JSON 1）** | 创建 `QuantizationSimModel` 时 | 是否量化、per-channel、对称性 | ONNX 算子名（如 `Conv`、`Gemm`、`Add`） |
| **阶段配置（JSON 2）** | `apply_mixed_precision_bitwidth` 时 | 各层位宽、输入/输出对称性 | AIMET 量化后类名（如 `QuantizedConv2d`、`QuantizedLinear`） |

`QuantGRU` 的位宽和内部算子配置位于阶段配置的 `GRU_config` 中。一次
`compute_encodings` 前向过程同时完成 QuantGRU 和其他模块的校准。
重建 QuantSim 时，`prepare_model` 参数、两份配置、默认位宽和量化方案必须与训练侧
一致；否则量化器名称或参数会错位。

### 5.4 已有 ONNX 模型的 PTQ

已有 ONNX 模型的 PTQ 入口是 `examples/onnx_ptq_quick_start.py`。该示例与
`quick_start_kws.py` 使用相同的步骤编号和 `apply_power_of_2_workflow` 实现，其余
步骤调用 `aimet_onnx.rx_ptq`。

```
加载 ONNX + 真实校准样本
  → QuantizationSimModel
  → set_percentile_value(99.99)
  → compute_encodings
  → cover_range Po2 + Sb = Sx * Sw
  → clean ONNX + compiler encodings
```

```bash
cp -a /opt/rx-met/examples /workspace/rx-met-examples
cd /workspace/rx-met-examples
python3 onnx_ptq_quick_start.py
```

命令退出码为 0，并输出“量化流程完成”以及 metadata 路径，即表示示例成功。本示例
使用 MobileNetV2。`examples/prepare_onnx_ptq_data.py` 将 Zoo 的动态 batch 模型
转换为静态 batch=1 模型，并按 ImageNet 预处理生成 calib/val npy。数据根目录默认
`/datasets/mobilenetv2`。换用其他模型时，通过
`RX_MET_ONNX_PTQ_MODEL` 和 `RX_MET_ONNX_PTQ_CALIB` 指定模型与校准目录。制品和限制
说明见 [`ONNX_PTQ_COMPILER.md`](./ONNX_PTQ_COMPILER.md)。

当前 ONNX 流程适用于 PTQ；QAT 使用 PyTorch 流程。编译器 encodings 是 rx-met 的
部署格式，必须与同次导出的 Clean ONNX 配套使用。

---

## 6. 常见问题（FAQ）

以下条目适用于 `1.0.0`，均为当前版本的已知排查方法。

### 6.1 镜像和运行环境

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `import quant_gru` 失败 | 镜像来源错误或镜像归档损坏 | 重新校验 `SHA256SUMS`，加载正确的 `rx-met:<版本>-cu*` 镜像 |
| `libpymo` 或 QuantGRU 报 CUDA 动态库缺失 | 镜像归档损坏，或容器中的 Torch/CUDA 包被替换 | 执行 `python3 -m pip check`，重新加载发布镜像并保留内置的 Torch/CUDA wheel |
| `torch.cuda.is_available()` 为 `False` | 容器缺少 GPU 映射，或宿主机驱动、Container Toolkit 异常 | 检查 `--gpus all`、宿主机 `nvidia-smi` 和 Container Toolkit，再确认镜像变体满足驱动要求 |

### 6.2 PyTorch 量化和重载

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 重载后精度与训练侧差异大 | 重建 QuantSim 使用了不同的模型预处理或量化配置 | 使用与训练侧相同的 `prepare_model` 参数、配置、默认位宽和量化方案 |
| `load_state_dict` 报缺失或多余键 | 量化模型包含额外的 quantizer 参数 | 使用 `load_state_dict(..., strict=False)`，再加载 encodings |
| 报量化参数未初始化 | 校准未覆盖已启用量化器，或加载时名称未匹配 | 检查校准数据路径和前向覆盖范围，并核对模型结构与配置文件 |

### 6.3 ONNX PTQ

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 找不到 `CUDAExecutionProvider` | ONNX Runtime、CUDA 变体或宿主机驱动存在兼容性差异 | 使用对应发布镜像；CPU 调试通过 `RX_MET_ONNX_PTQ_DEVICE=cpu` 启用 |
| 校准样本无法加载 | NPY/NPZ 的输入名、shape 或 dtype 与 ONNX 模型存在差异 | 根据模型输入重新生成校准数据，多输入模型使用同前缀 NPY 或单个 NPZ |

---

## 7. 卸载与回滚

镜像作为完整运行单元使用。回滚时加载上一个版本对应 CUDA 变体的
`tar.zst`，并切换完整镜像 tag。工作目录和数据通过 volume 挂载，其生命周期独立于
容器。
