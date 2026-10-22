# 公共环境构建材料

`third_party/` 保存构建公共 GPU 环境需要的离线第三方头文件。
两仓库各自保留这些材料，构建和使用镜像均不读取另一个仓库。

- `onnxruntime/` 保存 1.23.2 头文件，供 cu126 和 rx-met 本地 AIMET 编译使用。
- `onnxruntime-prepared/` 保存 cu118、cu130 对应的 1.20.1、1.27.0 头文件。

许可证、来源、版本和完整性清单随材料维护；公共 build 镜像统一把选中版本准备到
`/opt/rx-quant-env/onnxruntime`。runtime 镜像不包含这批构建头文件。

镜像通过环境版本标签复用，目录布局和构建脚本变化不会要求替换已有兼容镜像。
详见 [公共环境](../../docs/Shared_environment.md)。
