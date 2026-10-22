# ONNX Runtime 头文件

本目录头文件来自 ONNX Runtime `1.23.2`。公共头文件从官方
`onnxruntime-linux-x64-1.23.2` 发布归档提取，CUDA provider 头文件来自
`v1.23.2` tag：

- `include/core/providers/cuda/cuda_context.h`
- `include/core/providers/cuda/cuda_resource.h`

这些文件作为外部、不可变的构建输入归档在仓库中，支持 CPU/CUDA 原生库的离线
构建。编译前通过 `SHA256SUMS` 校验文件完整性。ONNX Runtime 使用 `LICENSE` 中的
MIT 许可证，第三方声明保留在 `ThirdPartyNotices.txt`。
