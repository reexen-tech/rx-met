# 默认量化配置

本目录保存运行时默认量化配置；机器可读格式契约位于算子根目录的 `schemas/`。

- `defaults/lstm_quant_default_v1.json` 是运行时默认值的唯一来源。
- `../schemas/lstm_quant_override.schema.json` 校验稀疏用户 override。
- `../schemas/lstm_quant_resolved.schema.json` 校验完整 resolved config。
- `../schemas/lstm_quant_params_bundle.schema.json` 校验 C++ 内部 canonical bundle。
- `../schemas/lstm_pytorch*_quant_params.schema.json` 校验公共参数交换文档。

Schema 和默认文件发生变化时，需要同步更新
[配置参考](../docs/configuration.md)、对应测试和[根目录 CHANGELOG](../../../CHANGELOG.md)。
