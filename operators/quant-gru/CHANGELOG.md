# QuantGRU 版本记录

本文档记录 QuantGRU 组件的版本、行为变化和迁移要求。组件版本来自
[pytorch/_version.py](pytorch/_version.py)。
既有版本日期沿用组件记录，详细开发历史保存在 Git 中。

## [Unreleased]

### 文档

- 按当前接口重写模块入口、配置说明和量化计算流程。
- 合并 16 段 PWL 原理，移除历史二次 LUT 草稿和重复算法文档。

### 修复

- 移除 CPU 参考构建中指向缺失 `example/test_runner.cc` 的目标，恢复默认 CMake 构建。

## [1.0.11] - 2026-07-16

### 变更

- MinMax、CPU/GPU 直方图校准统一通过 `encodeScaleResult()` 生成 scale。
- Po2 默认采用 `CoverRange`，容差为 2%。
- scale 编码集中到 `pot_scale_encode.h` 和 `scale_encoding.h`。
- Python binding 提供 Po2 策略和容差字段。

### 兼容性

Po2 默认取整行为发生变化。升级后应重新校准，并比较输出误差和保存重载结果。

## [1.0.10] - 2026-06-30

### 变更

- `GRUQuantParams` 使用 `QuantParam` 保存 scale 和零点，权重、偏置使用
  `ChannelQuantParam` 保存通道参数。
- 执行参数按 Po2、affine 和浮点缩放类型派生。
- Python binding 中权重、偏置 scale 统一为 `3 * hidden_size` 个通道元素。
- `use_pot2_scale` 默认值改为 `False`，对应 affine 模式。

### 修复

- 导出前同步因配置或校准数据变化而过期的量化参数。
- 参数过期且缺少校准数据时报告错误，提示重新校准。

### 兼容性

依赖 Po2 的配置显式设置 `use_pot2_scale=True`。读取旧 `scale_*_tensor_`、
`scale_*_gate_` 字段的调用方改用 `scale_W_`、`scale_R_`、`scale_bw_` 和
`scale_br_` 通道数组。

## [1.0.9] - 2026-06-26

### 修复

- Po2 导出保存实际的 `2^-n` scale，affine 导出保存连续校准 scale。
- 校准量化级数与整数上下界对齐，修正与 AIMET 相邻量化网格之间的 scale 差异。

## [1.0.8] - 2026-06-23

### 修复

- 量化级数使用 int64 计算，修复 32 位 bias 校准的整数溢出。
- Po2 移位量收窄到 int8 前增加范围检查，越界时报告错误。

## [1.0.7] - 2026-06-22

### 变更

- 单元素量化参数导出为标量，多元素参数导出为列表。
- ONNX 导出前通过 `ensure_quant_gru_onnx_registered()` 注册 opset，支持 opset 13
  及以上版本。

### 兼容性

符号注册使用的 opset 与 `torch.onnx.export(opset_version=...)` 保持一致。
下游解析器同时处理标量和列表形式的量化字段。

## [1.0.6] - 2026-06-12

### 新增

- `quant_storage_dtype` 提供 float32 和 int32 量化值存储路径。
- `forward_quantized()` 提供整数输入输出。
- `aimet_capabilities()`、`aimet_configure()` 和 `get_io_quant_meta()` 提供
  AIMET 集成与量化网格查询接口。

### 变更与限制

- 输入和隐藏状态的 MinMax 校准采用全局极值累计，双向执行共享输入网格。
- 隐藏状态校准从首个输出时间步开始，零初始状态为当前使用前提。
  有状态模型需要验证传入隐藏状态的范围覆盖。

## [1.0.5] - 2026-06-04

### 新增

- 支持连续 scale 和 affine 乘移执行表示。
- `export_mode` 通过 legacy exporter 导出标准 ONNX GRU 节点，支持单向和双向模型。

### 修复

- 修正 affine 边界量化、LUT scale 和乘子编码问题。
- 双向 AIMET encodings 按拼接后的 scale 长度生成对应零点。
- scale 模式切换后同步 Python 与 C++ 参数状态。

### 兼容性

量化导出字段统一为 `scale` 和 `zero_point`。依赖 `multiplier`、`shift`、
`scale_encoding` 的解析器迁移到新字段。加载接口继续解析历史参数格式。

## [1.0.4] - 2026-03-24

- 修复双向模型重载时反向输入量化参数的初始化。

## [1.0.3] - 2026-03-19

- 支持双向 AIMET encodings 导入导出。参数按正向、反向顺序拼接，反向激活参数保存
  在 `internal_ops_reverse`。

## [1.0.2] - 2026-03-13

- 增加 `set_quant_params_locked()`，控制后续校准对已有量化参数的覆盖。

## [1.0.1] - 2026-02-14

- 安装时将 `libgru_quant_shared.so` 复制到安装目录的 `lib/`，扩展通过
  `$ORIGIN/lib` 查找共享库。

## [1.0.0] - 2026-02-13

- 统一算子命名：`x/h` 改为 `input/output`，`W/R` 改为
  `weight_ih/weight_hh`，`bw/br` 改为 `bias_ih/bias_hh`。
- 移除当时用于 TF32 精度补偿的梯度缩放优化。

配置和接口调用采用[配置说明](pytorch/config/README.md)中的算子名称。
