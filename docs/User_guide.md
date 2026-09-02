# rx-met 用户使用指南

> 本文档遵循《Reexen 工具链软件文档管理规范》第 11.1 节「工具链快速上手」基线模板编写，
> 面向首次接触 rx-met 的用户，目标是让你在最短时间内跑通第一个量化流程。

| 项目 | 内容 |
|------|------|
| 文档名称 | rx-met 用户使用指南（快速上手） |
| 文档版本 | 2026.09-r1 |
| 适用软件版本 | 以发布包 `README.md` 中的组件版本和软件包 `vYYMMDD` 为准 |
| 文档责任人 | （待填写） |
| 最近更新 | 2026-09 |

相关文档：

- 量化配置详解：[`docs/Quant_config.md`](./Quant_config.md)
- 客户安装包：[`release/README.md`](../release/README.md)
- 安装包总览：[`README.md`](../README.md)

大模型量化不在本仓库。

---

## 1. rx-met 是什么

rx-met 是面向 ADA200 的**小模型量化工具包**：定制 AIMET（`aimet_torch` / `aimet_onnx` / `aimet_common`）+ QuantGRU。客户交付物是预编译 wheel，装进已有的 `ada200_docker`，**不含 Docker 镜像**。

| 工作流 | 适用对象 | 底层引擎 | 入口 | 对应章节 |
|--------|----------|----------|------|----------|
| **普通模型量化** | CNN / RNN / KWS 等中小模型（PyTorch） | `aimet_torch` | `examples/quick_start.py` | [第 5 章](#5-使用方法一普通模型) |
| **ONNX 直量化 PTQ** | 已有 ONNX、无 PyTorch 训练图 | `aimet_onnx` | `examples/onnx_ptq_quick_start.py` | [第 5.5 节](#55-已有-onnx-模型的-ptq) |

选择原则：

- 模型能在单卡显存内用 PyTorch 训练 / 推理，需要 **QAT、混合精度、Power-of-2 量化、ONNX 导出** → 走「普通模型」流程。
- 手里已经是 ONNX，只要 **PTQ + 编译器 encodings** → 走「ONNX 直量化 PTQ」。

---

## 2. 支持平台与版本要求

官方客户包对齐 `ada200_docker`：

| 项目 | 要求 |
|------|------|
| 操作系统 | Linux x86_64（Ubuntu 22.04） |
| Python | CPython 3.10 |
| PyTorch | `torch==2.8.0`，`torch.version.cuda == 12.8` |
| CUDA | 宿主机 NVIDIA Driver + NVIDIA Container Toolkit；容器内不需要再装 CUDA Toolkit |
| 核心依赖 | `numpy`、`scipy`、`onnx`、`onnxruntime`（镜像自带 CPU 版）、`pillow` |

`QuantGRU` 随发布包以 `quant_gru-*.whl` 安装，Haste GRU 模型可直接 `import quant_gru`。源码构建见 [`quant-gru-pytorch/`](../quant-gru-pytorch/)。

---

## 3. 安装

### 3.1 客户软件包（推荐）

把发布包挂进 `ada200_docker` 后：

```bash
cp -a /opt/rx-met /workspace/rx-met
cd /workspace/rx-met
./install.sh
```

脚本会校验 CPython 3.10、`torch==2.8.0+cu128`，再离线安装 `wheels/`。
详细步骤见 [`release/README.md`](../release/README.md)。

验证安装成功：

```bash
python -c "import aimet_torch, aimet_onnx, quant_gru; print('rx-met OK')"
```

### 3.2 仅安装 wheel

```bash
pip install --no-index --no-deps rx_met-*.whl quant_gru-*.whl
```

开发侧完整构建见根目录 [`README.md`](../README.md) 与 [`docs/Release_packaging.md`](./Release_packaging.md)。

---

## 4. 环境变量与路径配置

| 变量 / 路径 | 作用 | 适用流程 |
|-------------|------|----------|
| `RX_MET_SPEECH_COMMANDS_ROOT` | Speech Commands 数据集根目录，默认 `/datasets/speech_commands_v0.02` | 普通模型 |
| `RX_MET_ONNX_PTQ_ROOT` | ONNX 示例数据集根目录，默认 `/datasets` | ONNX PTQ |
| `RX_MET_ONNX_PTQ_EXAMPLE` | 示例名称，默认 `mobilenetv2` | ONNX PTQ |
| `RX_MET_ONNX_PTQ_MODEL` / `RX_MET_ONNX_PTQ_CALIB` | 覆盖模型与校准目录 | ONNX PTQ |
| `RX_MET_SKIP_ENV_CHECK` | `1` 时跳过 `install.sh` 环境检查 | 安装 |
| 运行目录 | demo 须 `cd examples/` 后运行 | 两者 |

---

## 5. 使用方法（一）：普通模型

本章基于 `examples/quick_start.py`（SpeechCommands + MRNN 关键词识别），演示 AIMET v2 的完整量化流程。该脚本的 `main()` 内联了所有标准 AIMET API，可直接照抄到自己的工程。

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
cd examples
export RX_MET_SPEECH_COMMANDS_ROOT=/datasets/speech_commands_v0.02
python quick_start.py
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

手里已经是 ONNX、不需要 QAT 时，用 `examples/onnx_ptq_quick_start.py`。步骤编号与 `quick_start.py` 对齐，Po2/bias 走同一个 `apply_power_of_2_workflow`；其余步骤导入 `aimet_onnx.rx_ptq`，不复制量化规则。

```
加载 ONNX + 真实校准样本
  → QuantizationSimModel
  → set_percentile_value(99.99)
  → compute_encodings
  → cover_range Po2 + Sb = Sx * Sw
  → clean ONNX + compiler encodings
```

```bash
cd examples
python onnx_ptq_quick_start.py
```

本示例使用 MobileNetV2。Zoo 模型是动态 batch，校准样本也不是 npy，需要先跑 `examples/prepare_onnx_ptq_data.py`（冻结 batch=1，按 ImageNet 预处理写出 calib/val npy）。数据根目录默认 `/datasets/mobilenetv2`。换自己的模型时，用 `RX_MET_ONNX_PTQ_MODEL` 和 `RX_MET_ONNX_PTQ_CALIB` 指向对应文件与校准目录。更细的编译器产物说明见 [`ONNX_PTQ_COMPILER.md`](./ONNX_PTQ_COMPILER.md)。

---

## 6. 常见问题（FAQ）

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| `import quant_gru` 失败 | 未安装发布包里的 QuantGRU wheel | 重新执行 `./install.sh`，或按 [`quant-gru-pytorch/`](../quant-gru-pytorch/) 从源码编译 |
| 重载后精度与训练侧差异大 | 重建 sim 的配置/`stateless_modules_to_preserve` 与训练时不一致 | 保证两侧完全一致（见 5.3 一致性要点） |
| `load_state_dict` 报缺失/多余键 | 量化模型含 quantizer 参数 | 使用 `load_state_dict(..., strict=False)` |
| `libpymo` 找不到 `libcudart.so.12` | 未登记 pip 自带的 NVIDIA CUDA runtime | `source cuda_libs.env`（`install.sh` 会生成） |
| `install.sh` 环境检查失败 | 不在 `ada200_docker`，或 torch 版本不对 | 使用官方镜像；临时跳过：`RX_MET_SKIP_ENV_CHECK=1 ./install.sh` |

---

## 7. 卸载与回滚

- **卸载 Python 包**：

```bash
pip uninstall rx-met quant-gru
```

- **回滚到旧版本**：安装对应日期的软件包即可，例如 `ada200-rx-met-vYYMMDD-linux_x86_64.tar.gz`。

---

## 8. 变更记录

| 版本 | 日期 | 变更项 | 责任人 |
|------|------|--------|--------|
| 2026.09-r1 | 2026-09 | 独立小模型仓：去掉大模型流程，对齐 ada200 wheel 交付 | （待填写） |
| 2026.06-r1 | 2026-06 | 首次发布：普通模型快速上手 | （待填写） |
