# rx-met 示例源码

本目录展示 rx-met 的完整使用流程。发布制品中的这份源码与产品镜像内
`/opt/rx-met/examples` 的内容一致，便于在加载镜像前审阅实现，也可以复制到
工作目录后修改。

示例应在同一批次交付的 rx-met 产品镜像中运行。模型、数据集和运行输出不包含在
源码目录中，应通过 volume 挂载或写入容器的工作目录。

## KWS 量化流程

`quick_start_kws.py` 使用 Speech Commands v0.02，依次执行浮点训练、PTQ、
Power-of-2、QAT、导出和重新加载验证：

```bash
export RX_MET_SPEECH_COMMANDS_ROOT=/datasets/speech_commands_v0.02
export RX_MET_KWS_OUTPUT_DIR=/workspace/output/quick_start_kws
python3 quick_start_kws.py
```

## ONNX PTQ 流程

`prepare_onnx_ptq_data.py` 将公开的 MobileNetV2 ONNX 和 ImageNette parquet 转换为
静态 batch=1 模型及 NPY 校准样本。`onnx_ptq_quick_start.py` 随后执行 PTQ、
Power-of-2、compiler encodings 导出和重新加载验证：

```bash
python3 prepare_onnx_ptq_data.py \
  --onnx /workspace/mobilenetv2-12.onnx \
  --parquet /workspace/imagenette2-320.parquet \
  --out-root /workspace/datasets/mobilenetv2

export RX_MET_ONNX_PTQ_ROOT=/workspace/datasets
export RX_MET_ONNX_PTQ_OUTPUT_DIR=/workspace/output/onnx_ptq
python3 onnx_ptq_quick_start.py
```

ONNX PTQ 默认要求 `CUDAExecutionProvider`。只有明确进行 CPU 调试时才设置：

```bash
export RX_MET_ONNX_PTQ_DEVICE=cpu
```

`config/` 保存两个流程共同使用的 QuantSim 和混合精度配置。
