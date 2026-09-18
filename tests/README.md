# 测试

本目录保存跨源码目录和构建配置的项目级测试。各模块内部的专项测试继续放在对应模块
目录中。

## 文件说明

`test_dependencies.py` 验证依赖管理配置和生成结果。该测试检查以下内容：

- `pyproject.toml` 的兼容范围覆盖全部发布变体。
- `docker/variants.json` 与生成的 Bake 配置和 lock 文件保持一致。
- 每个 CUDA 变体的运行环境锁完整且自包含。
- 构建工具锁与 manifest 保持一致。
- 缺失 Torch 必需包时配置校验会失败。
- wheelhouse checksum 兼容标准的相对路径格式。

该测试不访问网络，也不构建 Docker 镜像。修改依赖范围、变体矩阵、锁生成或
wheelhouse 校验逻辑后应重复执行：

```bash
python3 -m unittest -v tests/test_dependencies.py
```
