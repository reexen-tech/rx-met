# rx-met 用户使用指南

> 本文档遵循《Reexen 工具链软件文档管理规范》第 11.1 节「工具链快速上手」基线模板编写，
> 面向首次接触 rx-met 的用户，目标是让你在最短时间内跑通第一个量化流程。

| 项目 | 内容 |
|------|------|
| 文档名称 | rx-met 用户使用指南（快速上手） |
| 文档版本 | 2026.09-r2 |
| 适用软件版本 | `rx-met:1.0.0-cu118/cu126/cu130` |
| 文档责任人 | （待填写） |
| 最近更新 | 2026-09 |

相关文档：

- 量化配置详解：[`docs/Quant_config.md`](./Quant_config.md)
- 镜像使用说明：[`release/README.md`](../release/README.md)
- 项目总览：[`README.md`](../README.md)

大模型量化不在本仓库。

---

## 1. rx-met 是什么

rx-met 是面向 Linux x86_64 NVIDIA GPU 的**小模型量化工具包**：定制 AIMET（`aimet_torch` / `aimet_onnx` / `aimet_common`）+ QuantGRU。客户交付物是已经安装完整环境的独立 Docker 镜像。

| 工作流 | 适用对象 | 底层引擎 | 入口 | 对应章节 |
|--------|----------|----------|------|----------|
| **普通模型量化** | CNN / RNN / KWS 等中小模型（PyTorch） | `aimet_torch` | `examples/quick_start_kws.py` | [第 5 章](#5-使用方法一普通模型) |
| **ONNX 直量化 PTQ** | 已有 ONNX、无 PyTorch 训练图 | `aimet_onnx` | `examples/onnx_ptq_quick_start.py` | [第 5.5 节](#55-已有-onnx-模型的-ptq) |

选择原则：

- 模型能在单卡显存内用 PyTorch 训练 / 推理，需要 **QAT、混合精度、Power-of-2 量化、ONNX 导出** → 走「普通模型」流程。
- 手里已经是 ONNX，只要 **PTQ + 编译器 encodings** → 走「ONNX 直量化 PTQ」。

---

## 2. 支持平台与版本要求

三个镜像都是 Linux x86_64 GPU 镜像，但 CUDA、Python、Torch 和 ONNX Runtime
版本不同：

| 镜像变体 | 系统 / Python | CUDA | PyTorch | ONNX Runtime GPU | 严格最低 NVIDIA Driver |
|------|------|------|------|------|------|
| `cu118` | Ubuntu 22.04 / 3.10 | 11.8 | 2.7.1 | 1.20.1 | 520.61.05 |
| `cu126` | Ubuntu 22.04 / 3.10 | 12.6 | 2.8.0 | 1.23.2 | 560.35.05 |
| `cu130` | Ubuntu 24.04 / 3.12 | 13.0 | 2.10.0 | 1.27.0 | 580.126.20 |

宿主机需要 Docker、NVIDIA Driver 和 NVIDIA Container Toolkit；宿主机不需要
安装相同版本的 CUDA Toolkit。应按宿主机驱动和 GPU 计算能力选择变体，而不是
只看宿主机 `nvcc --version`。`cu130` 才包含 `sm_120` 目标；`cu118` 不支持
实验性 ExportedProgram API。

`QuantGRU` 和 CUDA 版 ONNX Runtime 已安装在镜像中，可直接导入。ONNX PTQ
默认要求 `CUDAExecutionProvider`，不会在 provider 缺失时静默回退 CPU。

---

## 3. 加载并启动镜像

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
共享盘；数据仍通过 `:ro` 挂载。`USER` 和 `LOGNAME` 用于兼容 PyTorch 的缓存
目录初始化，不会在镜像中创建宿主机用户。

验证环境：

```bash
python3 -c "import torch, aimet_torch, aimet_onnx, quant_gru; print(torch.__version__, torch.version.cuda)"
```

镜像内依赖已完整安装，不需要运行额外安装脚本。开发侧构建见
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

## 5. 使用方法（一）：普通模型

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

脚本会依次打印 8 个步骤的耗时与精度汇总（浮点 / PTQ / Po2 / QAT / 重载）。

### 5.3 关键步骤代码（照抄模板）

**步骤 1：准备模型并创建量化模拟器（基础配置）**

```python
import aimet_torch.v2 as aimet
from aimet_torch import model_preparer
from aimet_torch.v2 import quantsim
from aimet_torch.utils_rx import apply_mixed_precision_bitwidth

# prepare_model 把 forward 中的 functional 调用重构成可被 sim 抓取的 nn.Module
prepared_model = model_preparer.prepare_model(model)

sim = quantsim.QuantizationSimModel(
    prepared_model,
    dummy_input=dummy_input,
    quant_scheme="percentile",                 # min_max / tf / tf_enhanced / percentile
    config_file="config/mrnn_quantsim_config_custom_mixed_precision_v2.json",  # 基础配置
    default_output_bw=8,
    default_param_bw=8,
)
sim.set_percentile_value(99.99)                 # 仅 percentile 方案生效
```

**步骤 2：应用混合精度位宽（阶段配置，可选）**

```python
apply_mixed_precision_bitwidth(
    sim.model,
    config_file="config/quick_start_full_quant.json",  # 阶段配置：逐层指定位宽 / 对称性
    verbose=True,
)
```

**步骤 3：校准（PTQ）**

```python
with torch.no_grad(), aimet.nn.compute_encodings(sim.model):
    for idx, (x, _) in enumerate(calib_loader):
        if idx >= 100:        # 校准批次数
            break
        sim.model(x.to(device))
```

**步骤 4：Power-of-2 量化（NPU 友好，可选）**

```python
from aimet_torch.utils_rx import apply_power_of_2_workflow

apply_power_of_2_workflow(
    sim.model, method="round", tolerance=0.02,
    align_bias_scale=True, verbose=True,
)
```

**步骤 5：QAT 微调（可选）**

```python
from aimet_torch.utils_rx import freeze_quantizer_parameters, set_train_mode_freeze_bn

freeze_quantizer_parameters(sim.model, verbose=True, freeze_bn_affine=True)
# 之后用普通 PyTorch 训练循环微调，但用 set_train_mode_freeze_bn(sim.model) 替代 model.train()
```

**步骤 6：导出部署产物（ONNX + encodings）**

```python
from aimet_torch.rx_export.export_onnx_json import export_onnx_json

sim.model.cpu().eval()      # 导出要求模型在 CPU
onnx_path, enc_path = export_onnx_json(
    sim, export_dir="output", filename_prefix="my_model",
    dummy_input_shape=(1, 16000, 1), opset=18,
)
torch.save({"model": sim.model.state_dict()}, "output/my_model_qat.pth")
```

**步骤 7：重新加载验证（生产侧复现）**

```python
from aimet_torch.staged_quantization_utils import load_quantizer_encodings

# 用与训练时完全一致的参数重建 sim 后：
fresh_sim.model.load_state_dict(state, strict=False)   # 1. 加载权重
load_quantizer_encodings(fresh_sim.model, load_path=enc_path)  # 2. 加载量化参数
```

> ⚠️ **一致性要点**：重建 `sim` 时 `prepare_model` 的 `stateless_modules_to_preserve`、
> `config_file`、`default_*_bw`、量化方案必须与训练侧**完全一致**，否则 quantizer 名称错位会导致加载失败。

### 5.4 两类配置文件的区别

普通模型采用**双层配置**，详细字段说明见 [`Quant_config.md`](./Quant_config.md)：

| 配置 | 传入时机 | 控制内容 | 命名规则 |
|------|----------|----------|----------|
| **基础配置（JSON 1）** | 创建 `QuantizationSimModel` 时 | 是否量化、per-channel、对称性 | ONNX 算子名（如 `Conv`、`Gemm`、`Add`） |
| **阶段配置（JSON 2）** | `apply_mixed_precision_bitwidth` 时 | 各层位宽、输入/输出对称性 | AIMET 量化后类名（如 `QuantizedConv2d`、`QuantizedLinear`） |

> 进阶：当精度一次性全量化会崩溃时，可使用**多阶段量化**（逐步启用 Conv → GRU → BN…），
> 以及 **Haste GRU 专用配置 / 校准**，完整流程见 [`Quant_config.md`](./Quant_config.md) 第「多阶段量化配置」节。

### 5.5 已有 ONNX 模型的 PTQ

手里已经是 ONNX、不需要 QAT 时，用 `examples/onnx_ptq_quick_start.py`。步骤编号与 `quick_start_kws.py` 对齐，Po2/bias 走同一个 `apply_power_of_2_workflow`；其余步骤导入 `aimet_onnx.rx_ptq`，不复制量化规则。

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

本示例使用 MobileNetV2。Zoo 模型是动态 batch，校准样本也不是 npy，需要先跑 `examples/prepare_onnx_ptq_data.py`（冻结 batch=1，按 ImageNet 预处理写出 calib/val npy）。数据根目录默认 `/datasets/mobilenetv2`。换自己的模型时，用 `RX_MET_ONNX_PTQ_MODEL` 和 `RX_MET_ONNX_PTQ_CALIB` 指向对应文件与校准目录。更细的编译器产物说明见 [`ONNX_PTQ_COMPILER.md`](./ONNX_PTQ_COMPILER.md)。

---

## 6. 常见问题（FAQ）

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| `import quant_gru` 失败 | 使用的不是 rx-met 最终镜像，或镜像损坏 | 重新校验 `SHA256SUMS` 并加载正确的 `rx-met:<版本>-cu*` 镜像 |
| 重载后精度与训练侧差异大 | 重建 sim 的配置/`stateless_modules_to_preserve` 与训练时不一致 | 保证两侧完全一致（见 5.3 一致性要点） |
| `load_state_dict` 报缺失/多余键 | 量化模型含 quantizer 参数 | 使用 `load_state_dict(..., strict=False)` |
| `libpymo` 或 QuantGRU 报 CUDA 动态库缺失 | 镜像不完整，或绕过了镜像内 `ld.so` 配置 | 使用正式 rx-met 镜像并执行 `python3 -m pip check`；不要在容器里替换 Torch/CUDA wheel |
| `torch.cuda.is_available()` 为 `False` | 启动时未传 `--gpus all`，或宿主机驱动/Container Toolkit 不可用 | 检查 `nvidia-smi` 和 NVIDIA Container Toolkit，并选择满足驱动下限的变体 |

---

## 7. 卸载与回滚

镜像内环境是整体交付，不建议在容器中卸载或覆盖 Python 包。回滚时加载并运行
上一个版本对应 CUDA 变体的 `tar.zst`，并切换完整镜像 tag。工作目录和数据通过
volume 挂载，不随容器删除。

---

## 8. 变更记录

| 版本 | 日期 | 变更项 | 责任人 |
|------|------|--------|--------|
| 2026.09-r2 | 2026-09 | 改为 rx-met 自有 cu118/cu126/cu130 三镜像交付 | （待填写） |
| 2026.09-r1 | 2026-09 | 独立小模型仓：去掉大模型流程，对齐 ada200 wheel 交付 | （待填写） |
| 2026.06-r1 | 2026-06 | 首次发布：普通模型快速上手 | （待填写） |
