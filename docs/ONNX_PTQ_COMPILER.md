# ONNX PTQ compiler artifacts

这条流程直接读取已有 ONNX，不经过 `aimet_torch` 或
`aimet_torch.rx_export.export_onnx_json`。

名词（Clean ONNX / raw AIMET encodings / compiler encodings）见
[`docs/onnx-ptq-context.md`](../docs/onnx-ptq-context.md)。

照抄入口与 `examples/quick_start.py` 对应：

```bash
cd examples
python onnx_ptq_quick_start.py
```

## 运行前提

- CPython 3.10、Linux x86_64；
- 已安装 ONNX Runtime 1.23.2；
- 已构建 rx-met 的 ONNX native runtime；
- 校准目录使用真实数据，不使用随机数据生成正式产物。

## MobileNetV2

先用 `examples/prepare_onnx_ptq_data.py` 生成静态 batch=1 的 ONNX 和 calib npy：

```bash
python3 scripts/ptq_onnx_compiler.py \
  --model /datasets/mobilenetv2/mobilenetv2-12.onnx \
  --calib-dir /datasets/mobilenetv2/calib \
  --output-dir /datasets/mobilenetv2/output \
  --prefix mobilenetv2_ptq \
  --cpu
```

输入名从 ONNX 图自动读取。多输入模型可用同一 sample 前缀的 npy，或改用 npz。

## 输出

每次运行生成：

- `<prefix>.onnx`：移除 AIMET `QcQuantizeOp` 的 clean ONNX；
- `<prefix>.encodings`：compiler-native schema，包含 node-level `input/output`
  和 parameter encodings；
- `<prefix>.aimet.encodings`：AIMET 原始 encodings；
- `<prefix>.meta.json`：输入 shape、校准数量、Po2 变化、覆盖率和 FP32 对比。

默认对齐现有 Torch RX 量化契约：

- 校准：`percentile=99.99`（必须在 `compute_encodings` 前设置，native 默认 100 等于 min-max）
- 激活/权重对称量化；Conv/Gemm/MatMul 的 bias 不参与统计校准，校准后把 `Sb` 对齐到 `Sx * Sw`（默认 INT32）。AIMET ONNX 近零 bias 通道会得到 `scale=0`，校准期 fake-quant 会把后续激活打成 Inf/全 0。
- Power-of-2 使用与 `aimet_torch/power_of_2_quantization.py` 相同的 `cover_range`（容差 2%）
- 权重默认 per-channel；Gemm 保持 per-channel；无 Conv+Relu supergroup

配置与 `examples/quick_start.py` 相同：

- JSON 1：`examples/config/mrnn_quantsim_config_custom_mixed_precision_v2.json`
- JSON 2：`examples/config/quick_start_full_quant.json`（`Quantized*` / `GRU_config` 在当前 ONNX 图里没有则忽略）

`aimet_common/quantsim_config/onnx_ptq_compiler_w8a8.json` 仍可作为 `--config` 覆盖项。
