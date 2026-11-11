# 工具脚本

本目录保存环境、应用发布和模型制品验证入口。构建与交付命令在仓库根目录执行。

## 环境与应用发布

| 入口 | 用途 |
| --- | --- |
| `environment/build.sh`、`verify.sh` | 准备和验证公共 CPU/CUDA 环境；默认选择三个 CUDA 变体，CPU 显式指定 `cpu` |
| `environment/export_runtime.sh`、`export.sh` | 导出环境归档及各归档的校验、追踪信息；`export.sh` 可选择 build/runtime 角色 |
| `environment/load.sh` | 校验并导入指定环境归档 |
| `release/build.sh` | 构建应用包并离线安装自检；`--with-env` 可附带 runtime 归档 |
| `release/verify_bundle.sh` | 验收安装目录，CPU 检查 PTQ，CUDA 通过 `--gpu` 启用 GPU 验证 |

完整顺序和参数见[应用包发布](../docs/Release_packaging.md)与
[公共环境](../docs/Shared_environment.md)。

## 模型制品验收

`verify_compiler_artifacts.py` 只读检查 ONNX、schema v4 编码、张量维度和图映射，
默认要求 Po2 和完整计算输入覆盖：

```bash
python3 scripts/verify_compiler_artifacts.py \
  --onnx /path/to/model.onnx \
  --encodings /path/to/model.json \
  --report /path/to/acceptance.json
```

工具不修改输入文件，默认把 JSON 报告打印到标准输出。可选的 `--report` 同时写出报告文件，
可以覆盖已有报告。校验通过返回 0，制品校验失败返回 1；命令参数无效时由参数解析器报错。
`--allow-affine` 允许 affine scale，`--allow-partial` 只检查已声明的编码，不要求全部计算输入覆盖。

## 实现辅助模块

`environment/common.py` 维护环境定义与兼容检查，`runtime_archives.py` 处理归档。
`release/artifacts.py`、`verify_install.py`、`verify_runtime_archive.py` 实现发布与安装验收。
这些维护者实现不作为独立用户工具交付。
