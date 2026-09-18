# 项目文档

本目录保存用户指南、配置说明、维护者文档和架构决策。根目录 README 提供项目入口，
具体主题由以下文档展开。

## 用户文档

- [用户使用指南](./User_guide.md) 说明镜像加载、量化流程和常见问题。
- [量化配置](./Quant_config.md) 说明 QuantSim、混合精度和 QuantGRU 配置。
- [ONNX PTQ 编译器制品](./ONNX_PTQ_COMPILER.md) 说明已有 ONNX 模型的 PTQ 流程和输出格式。

## 维护者文档

- [Docker 构建与发布](./Release_packaging.md) 说明环境镜像和发布镜像的构建、验证与导出。
- [AIMET ONNX 原生运行库](./AIMET_ONNX_NATIVE.md) 说明原生运行库的来源、构建和验证。
- [参与贡献](../CONTRIBUTING.md) 说明修改、测试和提交要求。
- [版本记录](../CHANGELOG.md) 记录公开版本和待发布变更。

## 架构决策

`adr/` 保存已经接受的架构决策。ADR 记录设计理由和影响，并与用户指南、接口文档共同
描述系统行为。

用户可见行为发生变化时应更新对应指南。实现取舍发生变化时应补充或修订 ADR。
