# rx-met 镜像变更记录

## @VERSION@

- 交付方式改为项目自行维护的多 CUDA Docker 镜像。
- 新增 `cu118`、`cu126`、`cu130` 三个独立运行变体。
- 固定各变体的 Python、PyTorch 三件套、ONNX Runtime GPU、NVIDIA CUDA wheel
  和公共 Python 依赖。
- 产品镜像基于版本化 runtime-env 组装，项目源码与第三方依赖环境独立更新。
- 镜像内置定制 AIMET、QuantGRU、KWS quick start 和 ONNX PTQ quick start。
- 修复 ONNX quick start 对导出 encoding 字段名的错误读取。
