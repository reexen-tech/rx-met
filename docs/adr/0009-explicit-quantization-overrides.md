# 显式量化配置与旧编码的有效状态

- 状态：已接受
- 日期：2026-11-18

## 背景

ONNX JSON2 只修改基础配置已关闭量化器的 bitwidth，却不启用该点，与 Torch
补建输入/输出量化器的行为不同。另一方面，模型 I/O 默认规则又会覆盖显式禁用。
仅记录位宽或仅重载 enabled 编码不足以保持这些配置语义和排除层声明。

## 决策

显式非空的激活位宽请求启用有效的普通输入或输出量化点；默认回退、空值、单独
的对称性或符号请求不恢复量化。参数禁用、数学常量和 native/复合层的保护不由
该规则绕过。配置先形成共享量化器计划，冲突在修改前拒绝。

普通共享图输入使用消费者端口网格，避免一个层的启停反向影响另一个分支。
最终有效禁用以运行时对象身份跟踪，I/O 默认规则尊重它；持久化时转为稳定名字。

原始 AIMET 编码可选附加 `rx_quantization_overrides`，其中包含 schema_version=1、
disabled_quantizers 与 excluded_layers。编译器编码沿用 schema 4 的排除层字段。
源代码必须在加载任何 generic/native 网格前验证名字、共享对象及编码状态不矛盾，
成功后恢复覆盖状态。旧原始编码不被强制补此字段，实际加载启用的旧点会解除
目标模拟器残留的禁用或排除状态。

InstanceNormalization 的维度别名按实际节点输入 rank 筛选，允许 shape inference
确定中间输入 rank，不从外部输入或未填充的 ConnectedGraph shape 猜测。

LSTM 原生编码重载只归一已有的可推导表示：缺失 n 与已有的单元素 I/O 列表。
已有 n 及全部 scale、zero_point、dtype、参数值保留并验证；不迁移原生 offset
语义，也不通过删除严格比较或放宽数值容差兼容错误网格。

## 验证

使用真实 Torch/ONNX 配置对照、完整导出重载、共享别名矛盾元数据及加载前状态
不变回归。InstanceNorm 包含中间 Unsqueeze 输入和未知 rank；LSTM 同时验证
Torch/ONNX GRU/LSTM 四条加载路径，并确保坏请求不改变已有参数和锁状态。
