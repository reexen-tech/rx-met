# 使用示例

本目录保存 rx-met 的端到端量化示例。发布制品会将相同内容安装到
`/opt/rx-met/examples`。

## 文件说明

- `quick_start_kws.py` 演示 KWS 模型的浮点训练、PTQ、Power-of-2、QAT、导出和重载。
- `prepare_onnx_ptq_data.py` 将 ONNX 模型和 ImageNette 数据转换为静态模型及校准样本。
- `onnx_ptq_quick_start.py` 演示 ONNX PTQ、Power-of-2 和编译器制品导出。
- `config/` 保存 Torch 与 ONNX 示例共用的 QuantSim 和混合精度配置。

示例应在同一批次发布的产品镜像中运行。模型、数据集和输出通过 volume 挂载，不进入
源码目录或产品镜像。

KWS 示例使用以下入口：

```bash
export RX_MET_SPEECH_COMMANDS_ROOT=/datasets/speech_commands_v0.02
python3 quick_start_kws.py
```

ONNX PTQ 示例使用以下入口：

```bash
python3 prepare_onnx_ptq_data.py \
  --onnx /workspace/mobilenetv2-12.onnx \
  --parquet /workspace/imagenette2-320.parquet \
  --out-root /workspace/datasets/mobilenetv2
python3 onnx_ptq_quick_start.py
```
