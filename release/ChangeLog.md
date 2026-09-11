# rx-met 镜像变更记录

## @VERSION@

- 交付方式改为 rx-met 自有 Docker 镜像，不再依赖 ADA200 通用镜像。
- 新增 `cu118`、`cu126`、`cu130` 三个独立运行变体。
- 固定 Python 3.10、PyTorch 三件套、NVIDIA CUDA wheel 和公共 Python 依赖。
- 产品镜像基于版本化 runtime-env 组装，项目源码与第三方依赖环境独立更新。
- 镜像内置定制 AIMET、QuantGRU、KWS quick start 和 ONNX PTQ quick start。
- 修复 ONNX quick start 对导出 encoding 字段名的错误读取。
