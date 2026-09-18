# 使用示例

本目录保存 rx-met 的端到端量化示例。发布制品会将相同内容安装到
`/opt/rx-met/examples`。

示例在对应版本的 rx-met 发布镜像中运行。KWS 示例需要 Speech Commands v0.02；
ONNX PTQ 示例需要 ONNX 模型和代表性校准数据。

## 文件说明

- `quick_start_kws.py` 演示 KWS 模型的浮点训练、PTQ、Power-of-2、QAT、导出和重载。
- `prepare_onnx_ptq_data.py` 将 ONNX 模型和 ImageNette 数据转换为静态模型及校准样本。
- `onnx_ptq_quick_start.py` 演示 ONNX PTQ、Power-of-2 和编译器制品导出。
- `config/` 保存 Torch 与 ONNX 示例共用的 QuantSim 和混合精度配置。

模型、数据集和输出通过 volume 挂载，源码目录和产品镜像保持独立。

KWS 示例使用以下入口：

```bash
export RX_MET_SPEECH_COMMANDS_ROOT=/datasets/speech_commands_v0.02
python3 quick_start_kws.py
```

命令输出“量化流程完成（已通过加载验证）”并返回退出码 0，表示 KWS 示例完成。

ONNX PTQ 示例使用以下入口：

```bash
python3 prepare_onnx_ptq_data.py \
  --onnx /workspace/mobilenetv2-12.onnx \
  --parquet /workspace/imagenette2-320.parquet \
  --out-root /workspace/datasets/mobilenetv2
python3 onnx_ptq_quick_start.py
```

命令输出“量化流程完成”及 metadata 路径并返回退出码 0，表示 ONNX PTQ 示例完成。
详细环境变量和制品说明见 [`docs/User_guide.md`](../docs/User_guide.md) 与
[`docs/ONNX_PTQ_COMPILER.md`](../docs/ONNX_PTQ_COMPILER.md)。
