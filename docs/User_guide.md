# rx-met 用户使用指南

本文档说明如何安装应用包、启动匹配的 runtime，并运行 PyTorch 或 ONNX 量化流程。
文中的版本和命令以当前仓库为准。

相关文档：

- 量化配置详解：[`docs/Quant_config.md`](./Quant_config.md)
- 构建与发布：[`docs/Release_packaging.md`](./Release_packaging.md)
- 项目总览：[`README.md`](../README.md)

本仓库面向小模型量化。

---

## 1. rx-met 是什么

rx-met 是面向 Linux x86_64 NVIDIA GPU 的**小模型量化工具包**，包含定制
AIMET（`aimet_torch`、`aimet_onnx`、`aimet_common`）和 QuantGRU、QuantLSTM。项目通过独立
runtime 镜像提供第三方环境，项目代码由应用包安装后挂载。

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

CUDA 版 ONNX Runtime 安装在 runtime 中；QuantGRU 和 QuantLSTM 来自挂载的应用安装目录。ONNX PTQ
默认要求 `CUDAExecutionProvider`。

---

## 3. 加载并启动镜像

当前安装入口是应用包内的 `install.sh`。离线首次交付时，同时提供对应 runtime 归档，
放到包内或包旁边的 images/。从源码构建应用包见
[`Release_packaging.md`](./Release_packaging.md)。

```bash
./install.sh --offline
```

安装器打印应用目录和固定镜像 ID。将它们填入以下命令，并选择实际分配的 GPU、工作目录和数据目录：

```bash
RUNTIME_IMAGE_ID='sha256:替换为安装器输出的镜像ID'
docker run --gpus device=0 --rm -it \
  --ipc=host \
  --user "$(id -u):$(id -g)" \
  --group-add "$(stat -c '%g' /path/to/datasets)" \
  -e HOME=/tmp \
  -e USER="$(id -un)" \
  -e LOGNAME="$(id -un)" \
  -e PYTHONPATH=/opt/rx-met/python \
  -v /path/to/installed-version:/opt/rx-met:ro \
  -v /path/to/workspace:/workspace \
  -v /path/to/datasets:/datasets:ro \
  -w /workspace \
  "$RUNTIME_IMAGE_ID" bash
```

`--group-add` 让非 root 容器继承数据目录的只读组权限，适用于 CIFS/NFS
挂载存储；数据通过 `:ro` 挂载。容器身份由 `--user` 指定，`USER` 和 `LOGNAME`
用于初始化 PyTorch 缓存目录。

验证环境：

```bash
python3 -c "import torch, aimet_torch, aimet_onnx, quant_gru, quant_lstm; print(torch.__version__, torch.version.cuda)"
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

本章基于 `examples/quick_start_kws.py`（Speech Commands + QuantGRU / QuantLSTM 关键词识别），演示 AIMET v2 的完整量化流程。该脚本的 `main()` 内联了标准 AIMET API，可作为接入模板。

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
python3 quick_start_kws.py --rnn_type gru
# 选择 LSTM 时使用同一套流程和配置文件
python3 quick_start_kws.py --rnn_type lstm
```

脚本会依次打印 8 个步骤的耗时与精度汇总（浮点 / PTQ / Po2 / QAT / 重载）。命令
退出码为 0，并输出“量化流程完成（已通过加载验证）”，即表示示例成功。生成文件位于
`RX_MET_KWS_OUTPUT_DIR` 指定的根目录下，再按 `gru/`、`lstm/` 分开；未设置时使用
当前运行目录下的 `output/quick_start_kws/<gru|lstm>/`。也可使用 `--output-dir` 指定根目录。
默认各训练 1 epoch，仅用于学习流程。校准使用无随机增强的训练集，测试集只用于评估。
检查点记录网络类型，恢复时严格检查权重、同批样本输出与测试精度；失败返回非零退出码。
可独立阅读并随包交付的完整说明见 [examples/README.md](../examples/README.md)。

### 5.3 配置和一致性

完整 API 示例由 `examples/quick_start_kws.py` 维护，配置字段和最小代码片段见
[`Quant_config.md`](./Quant_config.md)。用户指南说明两类配置的职责边界：

普通模型采用**双层配置**，详细字段说明见 [`Quant_config.md`](./Quant_config.md)：

| 配置 | 传入时机 | 控制内容 | 命名规则 |
|------|----------|----------|----------|
| **基础配置（JSON 1）** | 创建 `QuantizationSimModel` 时 | 是否量化、per-channel、对称性 | ONNX 算子名（如 `Conv`、`Gemm`、`Add`） |
| **阶段配置（JSON 2）** | `apply_mixed_precision_bitwidth` 时 | 各层位宽、输入/输出对称性 | AIMET 量化后类名（如 `QuantizedConv2d`、`QuantizedLinear`） |

`QuantGRU` / `QuantLSTM` 的内部配置分别位于阶段配置的 `GRU_config` /
`LSTM_config` 中，普通层共用 `layer_type_config`。两种循环层配置可同时存在，
模型中的算子读取各自配置。一次 `compute_encodings` 前向同时完成循环层和普通层校准。
重建 QuantSim 时，`prepare_model` 参数、两份配置、默认位宽和量化方案必须与训练侧
一致；否则量化器名称或参数会错位。

`export_onnx_json()` 在所有图后处理和循环算子编码合并完成后，强制校验最终 ONNX
及 encodings，通过后才返回文件路径。校验以 `sim.model` 中启用的量化器为基准，检查：

- 输入、输出和权重量化编码是否完整，包括输入编号、每个通道及原生 GRU/LSTM 内部编码。
- 编码字段和值是否与模型一致，scale 是否为正数且有限。
- 最终 ONNX 的结构和 shape 是否有效、节点名是否重复，以及原生循环层节点是否匹配。
- 每个浮点计算节点的名称是否对应 activation encoding key；权重输入是否对应 initializer
  的 parameter encoding key。无法在最终图或简化前图中找到来源的编码 key 也会报错。
- 直接按 ONNX 输入检查覆盖率：Add/Mul/MatMul 等双浮点输入必须覆盖两路；单输入
  激活检查一路；Conv/Gemm 的激活和权重分别检查。即使量化器未启用，也不能绕过此检查。

缺少编码、缺少通道、编码值不一致或启用的张量量化器未校准时抛出 `RuntimeError`，
错误中列出模块及具体字段，例如 `encodings.activation_encodings.module_add.output.0`。
标量常量以及 shape/index 等非浮点输入无需编码；简化消除的节点可以保留有原始图来源
的编码。关闭量化的计算输入如果仍出现在最终浮点计算节点上，会因缺少编码而报错。
带偏置的高维 Linear 在导出副本中转换为 Reshape → Gemm → Reshape，保留融合计算
及原有输入、输出和参数编码；无偏置 Linear 保持单个 MatMul。Gemm 节点使用原模块
名称，权重与偏置名称保持对应，不新增中间量化位置，也不修改训练侧 QuantSim。
`prepare_model()` 自动将 `nn.GELU` 和 `F.gelu` 替换成显式计算模块，保留
`approximate="none"` 的 Erf 公式和 `approximate="tanh"` 的近似公式。展开后的
Div/Erf/Add/Mul/Pow/Tanh 按实际所选公式参与校准和 QAT，标量常量仍豁免。
必须在创建 QuantSim 前完成 prepare；已有 GELU 模型的浮点权重可复用，但原来的
模块边界编码不能覆盖新增中间量化器，需要重新校准并评估 QAT 精度。

KWS 示例使用 `QuantizableBatchNorm2d`，在 prepare 阶段展开为 Sub/Div/Mul/Add，
统计量通过算子输入参与校准；标准 BatchNorm 没有统计量编码时仍不能通过全部浮点输入
覆盖检查。导出器按 FX 参数使用位置将 reshape/常量折叠后的 initializer 恢复为参数编码
名称，不依赖参数数值猜测。乘一、加零和单输入 Concat 的量化边界保留在最终 ONNX。
这项校验不替代推理精度验证。
失败时已写出的文件仅用于排查，不应作为通过验证的部署产物使用。

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

以下条目适用于当前应用包与其匹配的公共 runtime。

### 6.1 镜像和运行环境

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `import quant_gru` 失败 | 应用挂载或 Python 搜索路径不正确 | 按应用包 README 检查只读挂载与 PYTHONPATH，并运行 install.sh 自检 |
| `libpymo` 或 QuantGRU 报 CUDA 动态库缺失 | 镜像归档损坏，或容器中的 Torch/CUDA 包被替换 | 执行 `python3 -m pip check`，使用应用包指定的公共 runtime 并保留其固定 Torch/CUDA 依赖 |
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

应用安装到独立版本目录。升级时挂载新目录并使用 manifest 指定的 runtime 启动新容器；
回滚时恢复旧目录与其匹配的 runtime。保留仍被容器使用的版本目录。
工作目录、数据和输出通过独立挂载保存。
