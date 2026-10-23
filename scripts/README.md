# 工具脚本

- environment/build.sh、verify.sh：准备和验证公共 GPU 环境。
- environment/export_runtime.sh、export.sh：导出公共 runtime 归档，生成独立归档校验和追踪信息。
- environment/load.sh：校验并导入公共 runtime。
- verify_compiler_artifacts.py：只读验收保存的 ONNX/schema v4 编码、张量维度及图映射，默认要求 Po2 和完整计算输入覆盖。
- environment/common.py：两仓库独立保存的公共环境定义；runtime_archives.py 处理独立归档的导出和加载。
- release/build.sh：默认只构建应用包并执行离线安装自检，输出 .release/export/rx-met-v<VERSION>/；--with-env 将运行环境归档导出或复用到平级环境目录。
- release/verify_bundle.sh：验证应用安装目录及 GPU 流程。
- release/artifacts.py、verify_install.py、verify_runtime_archive.py：维护者实现，不独立交付给用户。

详见 [发布说明](../docs/Release_packaging.md) 和 [公共环境](../docs/Shared_environment.md)。

```bash
python3 scripts/verify_compiler_artifacts.py \
  --onnx /path/to/model.onnx \
  --encodings /path/to/model.encodings \
  --report /path/to/acceptance.json
```

通过返回 0，失败返回 1；不修改输入文件。`--allow-affine` 允许 affine scale，
`--allow-partial` 只检查已声明的编码，不要求全部计算输入覆盖。
