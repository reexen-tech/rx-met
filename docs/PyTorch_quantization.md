# 接入自己的 PyTorch 模型

本文面向已有浮点模型、权重和代表性数据的使用者，按 PTQ、可选 QAT、导出与恢复的
顺序接入 RX-MET。先按[用户使用指南](User_guide.md)安装应用并进入匹配的 CPU/CUDA
runtime。复现项目现成模型时，直接使用[KWS 示例](../examples/README.md)。

下面的代码按顺序组成一个接入模板，面向**单个 Tensor 输入、输出分类 logits**的模型。
`my_project` 是用户自己的 Python 模块，不由 RX-MET 提供；完成模型、数据与检查点适配后
才能运行。多输入、多输出或其他任务需要相应调整前向调用、损失函数和评估指标。

## 1. 准备模型、权重与数据

项目侧需要提供以下对象：

| 对象 | 契约 |
| --- | --- |
| `build_model()` | 每次创建相同结构的浮点模型；原生循环层初始关闭量化 |
| `make_loaders()` | 返回 `train_loader, calibration_loader, validation_loader`；仅 PTQ 时训练 loader 可为 `None` |
| loader 的每个 batch | `(inputs, labels)`；本模板中 `inputs` 是模型实际输入的 Tensor，分类标签为整数类别编号 |
| 浮点检查点 | 本模板约定字典的 `model` 字段是 `build_model()` 对应的浮点 `state_dict` |

校准 loader 使用代表性数据和固定前处理，避免随机训练增强；验证集用于独立评估精度。
校准与验证 loader 必须非空、可重复遍历。MFCC、归一化等前处理应与实际部署一致，
并明确放在模型内还是模型外；外置前处理的参数需随主权重另行保存。
随机 Tensor 只适合检查接口和 shape，不能替代正式校准数据。

若要对 `nn.GRU` / `nn.LSTM` 的内部量化点使用原生算子，按
[循环算子接入](QuantLSTM_integration.md#使用与配置)显式替换；`prepare_model()` 不会
自动转换。可先将原始检查点加载到普通模型，再逐层替换并复制权重，或准备与
`build_model()` 所建结构匹配的检查点。验证替换前后的浮点输出后，再创建 QuantSim。

在容器的可写工作目录运行，确保 `my_project` 可导入：

```python
from pathlib import Path

import torch
import torch.nn.functional as F
import aimet_torch.v2 as aimet
from aimet_torch import model_preparer
from aimet_torch.v2 import quantsim
from aimet_torch.utils_rx import (
    apply_mixed_precision_bitwidth,
    apply_power_of_2_workflow,
    freeze_quantizer_parameters,
    set_train_mode_freeze_bn,
)
from aimet_torch.rx_export import export_onnx_json
from aimet_torch.staged_quantization_utils import load_quantizer_encodings

from my_project import build_model, make_loaders

device = torch.device("cuda")  # CPU PTQ 改为 "cpu"
fp_checkpoint = Path("/models/model_fp.pth")
output_root = Path("output/my_model")
config_dir = Path("/opt/rx-met/examples/config")
base_config = config_dir / "mrnn_quantsim_config_custom_mixed_precision_v2.json"
bitwidth_config = config_dir / "quick_start_full_quant.json"

train_loader, calibration_loader, validation_loader = make_loaders()
model = build_model().to(device).eval()
checkpoint = torch.load(fp_checkpoint, map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["model"], strict=True)
dummy_input = next(iter(calibration_loader))[0].to(device)
```

路径和 `checkpoint["model"]` 应按项目实际格式调整。在源码仓库运行时，将 `config_dir`
改为仓库内的 `examples/config`。应用目录只读；需要自定义配置时，先复制两份 JSON 到
工作目录，再改为使用副本路径。两份配置的职责、默认值和匹配优先级见[配置参考](Quant_config.md)。

本模板用分类 Top-1 作为任务指标；其他任务替换此函数：

```python
@torch.no_grad()
def evaluate(module):
    module.eval()
    correct, total = 0, 0
    for inputs, labels in validation_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = module(inputs)
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite validation output")
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.numel()
    if total == 0:
        raise ValueError("validation loader is empty")
    return correct / total

print("FP32:", evaluate(model))
```

## 2. 准备模型并创建 QuantSim

先在浮点模型上调用 `prepare_model()`，再创建模拟器并应用混合精度设置。
下面的辅助函数供首次量化和后续恢复共用，避免两侧配置不一致：

```python
def create_sim(prepared_model):
    sim = quantsim.QuantizationSimModel(
        prepared_model,
        dummy_input=dummy_input,
        quant_scheme="percentile",
        config_file=str(base_config),
        default_output_bw=8,
        default_param_bw=8,
    )
    sim.set_percentile_value(99.99)
    apply_mixed_precision_bitwidth(
        sim.model, config_file=str(bitwidth_config), verbose=True,
    )
    return sim

prepared = model_preparer.prepare_model(model)
sim = create_sim(prepared)
```

`dummy_input` 必须能走过实际前向路径。数据相关的 Python 控制流、自定义模块及部署图
覆盖问题需要按模型适配；不要通过跳过量化器来掩盖导出错误。调用
`apply_mixed_precision_bitwidth()` 后应确认所需的输入量化器存在，尤其是多输入算子的
第 2 路及后续输入。涉及 GELU、BatchNorm、
FX 参数及多输入模型时，先核对[图准备与导出约束](Export_and_reload.md#图准备与后处理)。

若项目另外调整了输入量化器或原生算子的开关，把这些操作也放入 `create_sim()`，
保证恢复时执行相同设置。

按单个算子调整位宽时，先查询本次 `sim.model` 的完整模块名并确认输入关系，再修改
项目 JSON 的 `layer_name_config`；名称查找、配置示例和生效校验见
[按单个 PyTorch 算子设置位宽](Quant_config.md#按单个-pytorch-算子设置位宽)。

## 3. 校准并评估 PTQ

```python
sim.model.to(device).eval()
with torch.no_grad(), aimet.nn.compute_encodings(sim.model):
    for inputs, _ in calibration_loader:
        sim.model(inputs.to(device))

print("PTQ:", evaluate(sim.model))
```

同一次前向校准普通层与原生 QuantGRU/QuantLSTM 内部量化点。
只被某些输入触发的分支也需要校准覆盖；启用但未校准的量化点可能在导出时被拒绝。
修改位宽、对称性、前处理或图结构后，应重新校准并评估；原生算子的配置重置限制见
[循环算子接入](QuantLSTM_integration.md#使用与配置)。

## 4. 按部署要求执行 Power-of-2

目标要求 Po2 scale 时，在校准后、参数冻结前转换，并重新评估。
以下用 `do_po2` 控制校准后的整模型转换；不要求这一步时设为 `False`。如果还希望
原生循环层从 affine 网格开始校准，请在加载配置前确认 `GRU_config.default_config.use_pot2_scale`
为 `false`，`LSTM_config.quant_config.scale_mode` 为 `"affine"`。本模板借用的 KWS
配置让 GRU 在初次校准时就使用 Po2。

```python
do_po2 = True
if do_po2:
    po2_report = apply_power_of_2_workflow(
        sim.model,
        align_bias_scale=True,
        bias_bitwidth=32,
        verbose=True,
    )
    print("Po2:", evaluate(sim.model))
```

这里显式为普通 Conv/Linear 启用 INT32 bias 乘积 scale 对齐；参数默认值和适用范围见
[校准和 Power-of-2](Quant_config.md#校准和-power-of-2)。原生 GRU/LSTM 使用自己的
bias 编码与重缩放规则，不能把普通层的对齐设置理解为自动改变循环层内部 bias。
Po2 和 bias 对齐可能改变精度，需要按目标后端要求和任务指标决定是否接受。

## 5. 可选：CUDA QAT

PTQ 达到要求时可直接进入导出。需要 QAT 时使用 CUDA，下面以一轮分类训练演示
调用顺序；轮数、学习率、损失和验证策略由项目决定：

```python
do_qat = False
if do_qat:
    if device.type != "cuda":
        raise ValueError("QAT requires CUDA")
    if train_loader is None:
        raise ValueError("QAT requires a training loader")
    freeze_quantizer_parameters(sim.model, freeze_bn_affine=True)
    optimizer = torch.optim.Adam(
        [parameter for parameter in sim.model.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    set_train_mode_freeze_bn(sim.model)
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(sim.model(inputs), labels)
        loss.backward()
        optimizer.step()
    print("QAT:", evaluate(sim.model))
```

每轮训练开始重新调用 `set_train_mode_freeze_bn()`；评估会将模型切回 eval。
QAT 沿用已校准的固定量化网格，不在每个训练 batch 重新校准。
CPU runtime 支持本教程的校准与 PTQ 路径，不支持浮点训练或 QAT。

## 6. 导出部署图并保存主权重

先保留同批验证样本的输出，再切到 CPU 导出。`export_onnx_json()` 返回实际制品路径；
导出器按默认策略创建独立运行目录，不能假设最终文件直接位于 `output_root`。

```python
sim.model.to(device).eval()
check_input = next(iter(validation_loader))[0].to(device)
with torch.no_grad():
    expected_output = sim.model(check_input).detach().cpu()

master_model = sim.get_original_model(sim.model, qdq_weights=False).cpu().eval()
sim.model.cpu().eval()
onnx_path, encodings_path = export_onnx_json(
    sim,
    export_dir=str(output_root),
    filename_prefix="model",
    dummy_input_shape=dummy_input.detach().cpu(),
    opset=18,
)
weights_path = Path(onnx_path).parent / "model.weights.pth"
torch.save({"model": master_model.state_dict()}, weights_path)
print("ONNX:", onnx_path)
print("Encodings:", encodings_path)
print("Master weights:", weights_path)
```

主权重文件是本教程显式保存的 PyTorch 检查点，与该次 ONNX/JSON 放在一起，
供后续恢复；部署使用同次导出的 ONNX 和 `.json`。
有外置前处理、网络变体等业务状态时，也要放入检查点并在恢复时校验。
导出严格检查不替代设备推理或任务精度验证，失败时已写出的文件只用于排查。
接口参数、编码格式、文件用途和图覆盖规则见[导出与恢复参考](Export_and_reload.md)。

## 7. 严格恢复并比较结果

这里保存的是准备后模型的主权重，因此先用相同方法准备新的模型、严格加载主权重，
再创建 QuantSim 并加载配套编码。`skip_if_not_found=False` 会把部分加载失败转为异常，
仍需检查加载统计和恢复结果；它不做反向覆盖检查，无法保证文件覆盖每个目标量化器。

```python
fresh_model = model_preparer.prepare_model(build_model().to(device).eval())
saved = torch.load(weights_path, map_location="cpu", weights_only=True)
fresh_model.load_state_dict(saved["model"], strict=True)
restored_sim = create_sim(fresh_model)
load_report = load_quantizer_encodings(
    restored_sim.model,
    load_path=encodings_path,
    skip_if_not_found=False,
    allow_overwrite=False,
    verbose=True,
)
print("Encoding load:", load_report)
restored_sim.model.to(device).eval()
with torch.no_grad():
    restored_output = restored_sim.model(check_input).detach().cpu()
torch.testing.assert_close(restored_output, expected_output, atol=1e-5, rtol=1e-4)
print("Restored:", evaluate(restored_sim.model))
```

上面的输出容差与项目 KWS 示例一致，其他模型需要制定适合其数值尺度的验收标准。
同时比较固定样本输出和独立验证集指标，记录 FP32、PTQ、Po2、可选 QAT、恢复后的结果。
本模板直接组合库 API；导出运行记录的 `success` 只表示该次导出结束，不包含后续的
主权重保存和恢复断言。完整接入应以脚本正常退出、恢复断言和任务指标共同判断。
若恢复失败，先核对模型结构、准备方式、两份配置、默认位宽和量化方案，再检查编码覆盖。
保存的是 QuantSim `state_dict` 或阶段文件时，按[对应恢复方式](Export_and_reload.md#权重和编码恢复)
处理，不与本教程的主权重检查点混用。
