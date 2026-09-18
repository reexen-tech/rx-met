# 项目文档

本目录保存用户指南、构建说明、架构决策和技术调研。根目录 README 只提供项目入口，
具体主题由本目录中的文档展开。

## 文档说明

- `User_guide.md` 说明镜像加载、量化流程、导出和常见问题。
- `Quant_config.md` 说明 QuantSim、混合精度和 QuantGRU 配置。
- `Release_packaging.md` 说明环境镜像和产品镜像的构建、验证与导出。
- `AIMET_ONNX_NATIVE.md` 说明 AIMET 原生运行库的来源、构建和验证。
- `ONNX_PTQ_COMPILER.md` 说明 ONNX PTQ 到编译器制品的流程。
- `onnx-ptq-context.md` 记录 ONNX PTQ 实现所需的上下文。
- `adr/` 保存已经接受的架构决策。
- `research/` 保存有来源引用的技术调研。

用户可见行为发生变化时应更新对应指南。实现取舍发生变化时应补充或修订 ADR。
