# GRU 独立 carrier 与统一激活开关

状态：接受。替代 [0005 中的共享整数前向决策](0005-gru-shared-integer-forward.md)，
保留其中参数锁定、网格恢复和 zero-point 修复。

`quant_storage_dtype` 的 CUDA 契约是选择量化码载体及其计算实现。将 FP32 入口转调
整数核心，再把缓存转换为 FP32，使原本的两条运行方式退化为只选择缓存类型。
恢复独立实现后，默认 `float32` 由 `quantGRUForwardFP()` 调用 `ForwardPassQuantFP`，
使用 FP32 量化码、SGEMM、浮点重缩放、显式舍入和截断；`int32` 由独立整数路径执行。
两者都实现量化后的数值流程，用户按目标设备、性能和精度需要选择。

所有量化边界统一为 `clamp(round_to_nearest_even(x / scale) + zero_point)`。
零点平移发生在舍入之后，避免奇数零点改变半格输入的最近偶数舍入结果。
调用者只使用完整的 `quantize()` / `quantize_f()`，一次完成舍入、零点平移、饱和、
安全存储类型转换与训练 mask 记录。编码 scale 的重载仅负责解码并转调完整入口。
未截断码值计算放在 `quant_gru_detail` 内部，不作为调用者需要组合的量化接口。

激活实现作为独立编译维度，由唯一的 `USE_REAL_ACTIVATION` 宏控制两条路径的三个门：

- 定义宏时，以实际量化 scale 反量化、计算标准 sigmoid/tanh、再量化。
- 未定义宏时，使用相同分段规则生成的 16 段 PWL；两条路径分别使用浮点或整数载体求值。

真实激活构建不需要 LUT，因而不包含 LUT 字段，也不执行 LUT 生成、初始化、复制和
设备参数传递。训练和推理不自行切换激活算法。标准 STE 反向继续使用前向门值与截断
mask，不改为 PWL 的分段斜率。

环境变量 `RX_MET_USE_REAL_ACTIVATION=0/1` 统一传给共享库、Python 扩展和 wheel 构建；
CMake option 为 `USE_REAL_ACTIVATION=OFF/ON`。默认关闭，仅开启时定义 C++ 宏。
显式 CMake 选项和已有缓存优先于环境变量；环境变量仅提供首次配置的默认值。
改变构建模式需要一致地重新编译共享库和扩展，不能混用不同模式的二进制。

验证覆盖两 carrier、两激活构建模式、affine/Po2、截断边界、非零 zero-point、QAT 和
编码重载，分别对照各自参考。独立执行允许浮点有效精度和乘移近似带来的数值差异，
不要求跨 carrier 输出、mask 和梯度逐位相同，也不据此推断任务精度提升或硬件加速。
FP32 对超过 `2^24` 的整数、乘积或累加不能保证精确；独立 NPU 对齐仍需目标端验证。

该恢复针对现有 CUDA 双路径。CPU 包目前仍只有整数量化核心，`float32` 仅改变返回的
码缓存类型；CPU 双载体独立实现需要单独设计和实现。
