# 工具脚本

- `release/build.sh`：使用已有固定环境，构建三个 wheel 应用包，并进行离线安装验收。
- `release/artifacts.py`：发布编排、runtime 导出、manifest 和安装器生成；研发内部使用。
- `release/verify_install.py`：在发布时内嵌进 install.sh 的安装自检，不作为独立文件交付。
- `release/verify_bundle.sh`：验证纯 runtime + 外部安装目录及可选 GPU 流程。
- `release/verify_kws_example.sh`：使用真实 Speech Commands 数据完整验证 GRU/LSTM KWS。
- `release/verify_onnx_ptq_example.sh`：使用真实模型和校准数据验证 ONNX PTQ。
- `environment/build.sh`、`verify.sh`：准备、验证独立版本的 build/runtime 环境。
- `environment/export_runtime.sh`：导出用户需要的 runtime-only gzip 归档。
- `environment/export.sh`、`load.sh`：内部 build/runtime 成对归档搬运。
- `internal/` 和 `lib/`：研发脚本辅助实现，不出现在应用包中。

详细使用见 [发布说明](../docs/Release_packaging.md)。
