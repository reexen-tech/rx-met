# rx-met 用户使用指南

本文档面向使用应用包的用户，说明平台选择、安装、容器启动和量化任务入口。
应用包已经提供项目 Python 包和原生算子，无需另行编译。完整文档分类见[文档索引](README.md)。
维护者从源码构建环境和应用包时，阅读[构建与发布](Release_packaging.md)。

---

## 1. rx-met 是什么

rx-met 是面向 Linux x86_64 CPU 和 NVIDIA GPU 的**小模型量化工具包**，包含定制
AIMET（`aimet_torch`、`aimet_onnx`、`aimet_common`）和 QuantGRU、QuantLSTM。项目通过独立
runtime 镜像提供第三方环境，项目代码由应用包安装后挂载。

| 工作流 | 适用对象 | 底层引擎 | 入口 | 对应章节 |
|--------|----------|----------|------|----------|
| **PyTorch 模型量化** | 已有浮点模型、权重和校准数据 | `aimet_torch` | [自有模型接入教程](PyTorch_quantization.md)；KWS 为可运行实例 | [第 5.1 节](#51-完整流程总览) |
| **ONNX 直量化 PTQ** | 已有 ONNX、无 PyTorch 训练图 | `aimet_onnx` | `examples/quick_start_onnx_ptq.py` | [第 5.4 节](#54-已有-onnx-模型的-ptq) |

选择原则：

- 已有 PyTorch 模型、权重和代表性数据 → 走「PyTorch 模型量化」流程；CPU 可用于 PTQ，需要浮点训练或 QAT 时使用 CUDA。
- 已有 ONNX 模型需要 **PTQ + 编译器 encodings** → 走「ONNX 直量化 PTQ」。

---

## 2. 支持平台与版本要求

runtime 提供一个 CPU 变体和三个 CUDA 变体，均面向 Linux x86_64。
CPU 支持校准与 PTQ；浮点训练和 QAT 使用 CUDA。各变体的依赖如下：

| 镜像变体 | 系统 / Python | CUDA | PyTorch | ONNX Runtime | 严格最低 NVIDIA Driver |
|------|------|------|------|------|------|
| `cpu` | Ubuntu 22.04 / 3.10 | 无 | 2.9.1 CPU | 1.23.2 CPU | 无 |
| `cu118` | Ubuntu 22.04 / 3.10 | 11.8 | 2.7.1 | 1.20.1 | 520.61.05 |
| `cu126` | Ubuntu 22.04 / 3.10 | 12.6 | 2.8.0 | 1.23.2 | 560.35.05 |
| `cu130` | Ubuntu 24.04 / 3.12 | 13.0 | 2.10.0 | 1.27.0 | 580.126.20 |

宿主机需要 Bash、GNU coreutils、GNU findutils、`flock`（util-linux），以及通过 Unix socket
访问本机 daemon 的 Docker CLI；安装器不支持远程 Docker context 或 TCP/SSH
`DOCKER_HOST`。加载 runtime 归档时还需要 `gzip`，在线下载归档时另需 `curl`。
离线安装不需要宿主机 Python 或编译器。
CUDA 变体还需要 NVIDIA Driver 和 NVIDIA Container Toolkit。
CUDA 镜像已经包含对应版本的 CUDA 运行环境，变体由宿主机驱动版本和 GPU 计算能力共同决定；宿主机
`nvcc --version` 显示本地 CUDA Toolkit 版本。`cu130` 包含 `sm_120` 目标；
实验性 ExportedProgram API 适用于 `cu126` 和 `cu130`，与本指南所用的 legacy ONNX
导出接口不同；普通量化流程无需使用它。

CPU runtime 安装 `onnxruntime`，CUDA runtime 安装 `onnxruntime-gpu`；QuantGRU 和
QuantLSTM 来自挂载的应用安装目录。ONNX PTQ 示例默认要求 `CUDAExecutionProvider`，
使用 CPU 时显式指定 `--device cpu` 或 `RX_MET_ONNX_PTQ_DEVICE=cpu`。
精确依赖及 GPU 架构以 [docker/variants.json](../docker/variants.json) 为准。

---

## 3. 加载并启动镜像

在宿主机解压所选应用包，进入解压目录后执行 `install.sh`。本机 Docker 已有匹配镜像时，
安装器会复用它。首次离线安装且缺少镜像时，将 `rx-quant-env-<环境版本>/` 公共运行环境
目录与存放应用包的 `rx-met-v<版本>/` 产品目录平级放置，或通过 `--runtime-image FILE`
指定镜像归档。环境与应用包的对应关系见[公共环境](Shared_environment.md)。

```bash
./install.sh --offline
```

安装器打印应用目录和公共环境版本标签。先在宿主机创建工作目录，并准备模型与数据目录。
将实际路径和镜像标签填入以下 CUDA 容器命令；CPU 应用包省略 `--gpus device=0`：

```bash
RUNTIME_IMAGE='替换为安装器启动命令中的镜像版本标签'
docker run --gpus device=0 --rm -it --pull=never \
  --shm-size=2g \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e USER="$(id -un)" \
  -e LOGNAME="$(id -un)" \
  -e PYTHONPATH=/opt/rx-met/python \
  -v /path/to/installed-version:/opt/rx-met:ro \
  -v /path/to/workspace:/workspace \
  -v /path/to/models:/models:ro \
  -v /path/to/datasets:/datasets:ro \
  -w /workspace \
  "$RUNTIME_IMAGE" bash
```

模型和数据以 `:ro` 挂载；应用目录也保持只读，工作目录用于保存代码和输出。
容器身份由 `--user` 指定，`USER` 和 `LOGNAME` 用于初始化 PyTorch 缓存目录。
若 CIFS/NFS 数据需要额外的组权限，可在同一条 Docker 命令中加入
`--group-add "$(stat -c '%g' /path/to/datasets)"`，让非 root 容器继承数据目录的组权限。

验证环境：

```bash
python3 -c "import torch, aimet_torch, aimet_onnx, quant_gru, quant_lstm; print(torch.__version__, torch.version.cuda)"
```

命令退出码为 0，并打印与所选镜像匹配的 PyTorch 和 CUDA 版本，即表示基础环境可用；
CPU runtime 的 CUDA 版本打印为 `None`。runtime 提供第三方依赖，应用挂载提供项目包。

---

## 4. 环境变量与路径配置

| 变量 / 路径 | 作用 | 适用流程 |
|-------------|------|----------|
| `RX_MET_SPEECH_COMMANDS_ROOT` | Speech Commands 数据集根目录，默认 `/datasets/speech_commands_v0.02` | KWS 示例 |
| `RX_MET_KWS_OUTPUT_DIR` | KWS 输出根目录，默认 `./output/quick_start_kws`，自动追加网络类型和运行 ID | KWS 示例 |
| `RX_MET_KWS_FP_MODEL` | 浮点检查点路径；`--ptq-only` 时必须由此变量或 `--fp-checkpoint` 提供 | KWS 示例 |
| `RX_MET_ONNX_PTQ_ROOT` | 校准数据根目录，默认 `./output/onnx_ptq_quick_start/datasets` | ONNX PTQ |
| `RX_MET_ONNX_PTQ_EXAMPLE` | 示例名称，默认 `mobilenetv2` | ONNX PTQ |
| `RX_MET_ONNX_PTQ_MODEL` / `RX_MET_ONNX_PTQ_CALIB` | 覆盖模型与校准目录 | ONNX PTQ |
| `RX_MET_ONNX_PTQ_VAL` | 可选验证目录；默认查找校准目录的平级 `val/`，不存在时复用校准样本作数值比较 | ONNX PTQ |
| `RX_MET_ONNX_PTQ_OUTPUT_DIR` | 输出目录，默认当前运行目录下的 `output/onnx_ptq_quick_start` | ONNX PTQ |
| `RX_MET_ONNX_PTQ_DEVICE` | `cuda`（默认）或显式设置为 `cpu` | ONNX PTQ |
| 镜像内示例目录 | `/opt/rx-met/examples` | 两者 |
| 推荐输出目录 | volume 挂载的 `/workspace` | 两者 |

---

## 5. 量化流程

环境可用后，按下面的任务选择入口。示例的数据准备和命令由
[示例说明](../examples/README.md)维护；自有模型的接入步骤见
[PyTorch 模型量化](PyTorch_quantization.md)。

### 5.1 完整流程总览

```
已有浮点模型与权重 → prepare_model → QuantizationSimModel + 混合精度位宽
  → compute_encodings 校准 (PTQ)
  → apply_power_of_2_workflow (NPU 友好的 Po2 量化)
  → 可选：freeze_quantizer_parameters + CUDA QAT 微调
  → 分别保存主权重检查点、ONNX 和 .json 编码文件
  → 重建并加载主权重 → 创建 sim → load_quantizer_encodings → 精度对比
```

已有权重可直接进行 PTQ；浮点训练不是量化前必须重复执行的步骤。KWS 完整示例同时演示
浮点训练与 QAT，CPU 模式使用已有检查点，只执行校准、PTQ、Po2、导出与恢复。
按[自有模型接入教程](PyTorch_quantization.md)准备模型、数据和评估函数，
再根据[配置参考](Quant_config.md)调整量化设置。

### 5.2 运行示例脚本

在容器内将示例复制到可写工作目录；已按发布包 README 在宿主机复制过示例时，直接进入
`/workspace`。以下路径与[示例说明](../examples/README.md)一致：

```bash
cp -a /opt/rx-met/examples /workspace/
cd /workspace
```

| 任务 | 输入与运行说明 |
| --- | --- |
| CUDA 上运行完整 KWS GRU/LSTM 流程 | [KWS 示例](../examples/README.md#kwsgru--lstm)，需要 Speech Commands v0.02 |
| CPU/CUDA 上只做 KWS 校准与 PTQ | [固定检查点 PTQ](../examples/README.md#只校准和-ptqcpu--gpu)，需要同类型浮点检查点及数据 |
| ONNX PTQ 公开演示 | [MobileNetV2 两步流程](../examples/README.md#onnx-ptqmobilenetv2)，先准备模型和校准样本，再量化 |

KWS 默认浮点训练和 QAT 各 1 epoch，用于学习流程，不能据此推断充分训练后的精度。
示例验证重载输出，失败返回非零；产物位置和日志说明见[运行目录与日志](Runtime_outputs.md)。

### 5.3 配置和一致性

PyTorch 和 ONNX 采用两层配置。下表以 PyTorch 调用为例；完整字段、匹配优先级和校准后的
修改限制见[PyTorch 配置参考](Quant_config.md)及 [ONNX PTQ 配置说明](ONNX_PTQ_COMPILER.md)。

| 配置 | 传入时机 | 控制内容 | 命名规则 |
|------|----------|----------|----------|
| **基础配置（JSON 1）** | 创建 `QuantizationSimModel` 时 | 是否量化、per-channel、对称性 | ONNX 算子名（如 `Conv`、`Gemm`、`Add`） |
| **混合精度配置（JSON 2）** | `apply_mixed_precision_bitwidth` 时 | 普通层位宽与整数网格、循环层配置 | AIMET 量化后类名或完整模块名；循环层用 `GRU_config` / `LSTM_config` |

`QuantGRU` / `QuantLSTM` 的内部配置分别位于混合精度配置的 `GRU_config` /
`LSTM_config` 中，普通层共用 `layer_type_config`。两种循环层配置可同时存在，
模型中的算子读取各自配置。一次 `compute_encodings` 前向同时完成循环层和普通层校准。
重建 QuantSim 时，`prepare_model` 参数、两份配置、默认位宽和量化方案必须与训练侧
一致；否则量化器名称或参数会错位。

导出、重载、多输入模型和图准备约束见[导出与恢复参考](Export_and_reload.md)。
部署时使用同次导出返回的 ONNX 和 `.json` 配套路径，严格校验成功后再进行精度验证；
失败时已写出的文件仅用于排查。

### 5.4 已有 ONNX 模型的 PTQ

已有 ONNX 模型使用 `examples/quick_start_onnx_ptq.py`，无需先转换为 PyTorch。
准备静态模型与代表性的 NPY/NPZ 校准样本，建议另备独立验证样本，按
[ONNX PTQ 文档](ONNX_PTQ_COMPILER.md)运行；输入名、shape 和 dtype 必须与模型一致。

```
加载 ONNX + 真实校准样本
  → QuantizationSimModel
  → set_percentile_value(99.99)
  → compute_encodings
  → cover_range Po2 + Sb = Sx * Sw
  → clean ONNX + compiler encodings
```

CPU runtime 显式传入 `--device cpu`。公开 MobileNetV2 演示的下载、预处理和量化命令见
[示例说明](../examples/README.md#onnx-ptqmobilenetv2)；通用入口不负责用户模型的预处理。

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
| CUDA runtime 中 `torch.cuda.is_available()` 为 `False` | 容器缺少 GPU 映射，或宿主机驱动、Container Toolkit 异常 | 检查 `--gpus all`、宿主机 `nvidia-smi` 和 Container Toolkit，再确认镜像变体满足驱动要求；CPU runtime 返回 `False` 为正常情况 |

### 6.2 PyTorch 量化和重载

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 重载后精度与训练侧差异大 | 重建 QuantSim 使用了不同的模型预处理或量化配置 | 使用与训练侧相同的 `prepare_model` 参数、配置、默认位宽和量化方案 |
| `load_state_dict` 报缺失或多余键 | 模型结构、准备方式或检查点类型不匹配 | KWS 检查点保存主权重；先用相同网络类型和准备方式严格加载，再创建 QuantSim 并加载 encodings。不要用 `strict=False` 隐藏结构差异 |
| 报量化参数未初始化 | 校准未覆盖已启用量化器，或加载时名称未匹配 | 检查校准数据路径和前向覆盖范围，并核对模型结构与配置文件 |

### 6.3 ONNX PTQ

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 找不到 `CUDAExecutionProvider` | 选用了 CPU runtime，或 CUDA 运行环境不兼容 | CPU PTQ 使用 `--device cpu` 或 `RX_MET_ONNX_PTQ_DEVICE=cpu`；CUDA PTQ 检查匹配的 runtime 和宿主机驱动 |
| 校准样本无法加载 | NPY/NPZ 的输入名、shape、dtype 或数值不符合模型要求 | 按 [ONNX 输入要求](ONNX_PTQ_COMPILER.md#量化自己的-onnx-模型)生成数据；多输入推荐每份样本保存一个以完整输入名为键的 NPZ |

---

## 7. 卸载与回滚

应用安装到独立版本目录。升级时挂载新目录并使用 manifest 指定的 runtime 启动新容器；
回滚时恢复旧目录与其匹配的 runtime。保留仍被容器使用的版本目录。
工作目录、数据和输出通过独立挂载保存。
