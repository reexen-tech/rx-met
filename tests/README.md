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

该测试执行本地依赖配置一致性检查。网络下载和 Docker 镜像构建由对应发布验证流程
覆盖。修改依赖范围、变体矩阵、锁生成或 wheelhouse 校验逻辑后执行：

```bash
python3 -m unittest -v tests/test_dependencies.py
```


## 循环算子与客户示例

`test_quant_lstm_integration.py` 使用随机小模型验证 LSTM 的校准、实际 QAT 更新、
参数锁、Po2、GRU/LSTM 共存和 ONNX 编码。`test_kws_example.py` 调用客户 KWS
主入口，使用小尺寸网络和随机音频验证 GRU/LSTM 两种模式的完整流程、导出节点及
检查点保护；另检查校准数据来自训练集且关闭增强。随机数据不用于评价任务精度。

在安装了 rx-met、QuantGRU、QuantLSTM 的 CUDA 环境中执行：

```bash
python3 -m unittest -v tests/test_quant_lstm_integration.py tests/test_kws_example.py
```

面向发布镜像的真实 Speech Commands 数据验收使用
`scripts/release/verify_kws_example.sh`，默认分别运行 GRU 和 LSTM。
