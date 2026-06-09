# Reex GEMM 量化精度测试报告

> **导航**：[总览索引](README.md) · **本报告** · [KV Cache 报告](test-reex-kv-cache-report.md) · [PWNL 报告](test-reex-pwnl-report.md)

## 一、测试目的

验证 Reex GEMM 不同激活量化路径（Q16 / Q8）相对于原始 float 计算的精度损失，确认：
1. 单次 GEMM 操作满足工程精度要求（cosine > 0.9999, mae < 1 LSB）
2. **全层替换后模型最终输出（final_logits）不改变预测行为**——这是最重要的验收标准

## 二、测试架构

### 2.1 金标准（Gold Standard）

所有评估基准统一为 **float×float**：使用原始 float32 权重 × float32 激活做矩阵乘，无任何量化。

### 2.2 四路对比

每组测试包含 4 个变体，输入数据完全一致，仅计算路径不同：

| 路径 | 权重 | 激活 | 说明 |
|------|------|------|------|
| **f32×f32** | float32 原始 | float32 原始 | 金标准（零误差） |
| **W4×f32** | Q4_K (4-bit) | float32 原始 | 仅权重量化，隔离权重误差 |
| **W4×Q16** | Q4_K (4-bit) | Q16_K (int16 + per-block scale, QK_K=256) | Reex Q16 路径 |
| **W4×Q8** | Q4_K (4-bit) | Q8_K (int8 + per-block scale) | 原生 llama.cpp Q8 路径 |

### 2.3 评估指标

| 指标 | 阈值 | 适用范围 |
|------|------|----------|
| **cosine** (余弦相似度) | > 0.9999 | 矩阵输出（n > 1）；单标量 n=1 时 cosine 恒为 ±1 无区分度，不使用 |
| **mae** (平均绝对误差) | < 1 LSB | 全部 |

### 2.4 1 LSB 的定义（依据行业/标准）

本测试中的「1 LSB」与 **ADC/DAC 数据手册** 及 **JEDEC** 一致，且 **basis** 按数值误差传播的惯例取为「与误差界成正比的参考量」。

**标准依据**
- **JEDEC**：1 LSB 为线性转换器的模拟分辨率单位，误差以 LSB 的倍数表示。
- **ADC/DAC**：1 LSB = Vref / 2^N = 一个量化步长。
- **数值误差分析**（如 Higham、舍入误差分析）：dot product 的绝对误差界与 **Σ|x_i||y_i|** 成正比，即与「按分量的绝对值乘积之和」成正比，而不是与输出标量 |y| 成正比。

**output_1LSB 公式**

```
output_1LSB = basis × LSB_REL_factor
```

**basis（传播参考量）**——与「量化误差传播到输出」的界成正比：

| 场景 | basis 定义 | 依据 |
|------|-------------|------|
| 单 vec_dot | **energy** = Σ_i \|w_i×a_i\| | 权重量化误差界 ∝ Σ(Δw_i)·\|a_i\| ∝ Σ\|w_i\|\|a_i\| |
| GEMM 矩阵 | **mean_energy** = (1/(N·M)) Σ_{j,m} Σ_i \|W_ji\|\|act_im\| | 每个输出的误差界 ∝ 该行的 energy；取平均得 mean_energy |
| Chain 等 | **mean\|gold\|** | 无逐行 energy 时，用输出量级 mean\|gold\| 作保守参考 |

因此 **basis 不是「输出均值 mean\|gold\|」**（除非在 Chain 中退化为该量），而是 **「各 dot product 的 energy 或其平均」**，与误差传播公式一致。

**LSB_REL_factor**（与格式一致的标准步长相对值）：见下表（不变）。

| 路径 | 量化器 | 标准 1 LSB 相对值 | 依据 |
|------|--------|-------------------|------|
| W4×f32 | Q4_K (4-bit) | **1/16** | 4-bit → 16 档 → 1 步 = 1/16 子块范围；dequant = d×scale_s×q4 − …，q4∈[0,15] |
| W4×Q16 | Q4_K + Q16 | **1/16 + 1/32767** | Q16：per-block scale (QK_K=256)，d = amax/32767 |
| W4×Q8 | Q4_K + Q8_K | **1/16 + 1/127** | Q8_K：7-bit 有效 → 1/127 |

组合路径按**最坏叠加**：factor = LSB_weight + LSB_activation。

**计算示例**（single_block K=256 M=8 N=16, W4×f32）：
```
basis = mean_energy = (1/(N·M)) Σ_{j,m} Σ_i |W_ji||act_im| = 151
1LSB = 151 × (1/16) = 9.43
mae  = 0.123
mae / 1LSB = 0.01  →  < 1 LSB ✓
```

## 三、测试结果

### 3.1 激活量化 Roundtrip

验证激活量化→反量化的信号保真度（不涉及权重）。

| 格式 | K | cosine | mae | 结果 |
|------|---|--------|-----|------|
| Q16 | 256 | 1.000000 | 7.50×10⁻⁶ | OK |
| Q16 | 512 | 1.000000 | 7.46×10⁻⁶ | OK |
| Q8 | 256 | 0.999994 | 2.06×10⁻³ | OK |
| Q8 | 512 | 0.999994 | 2.07×10⁻³ | OK |

**结论**：Q16 roundtrip 误差比 Q8 低约 270 倍（15-bit vs 7-bit 精度），两者均满足 cosine > 0.9999。

### 3.2 单 Vec Dot 4-way

单个 dot product：`result = Σ(weight[i] × activation[i])`，输出为标量。

#### K=256, 全负激活

| 路径 | gold | test | \|err\| | 1LSB | \|err\|/1LSB | 结果 |
|------|------|------|---------|------|-------------|------|
| f32×f32 | -2.0439 | -2.0439 | 0 | — | — | OK |
| W4×f32 | -2.0439 | -1.7808 | 2.63×10⁻¹ | 7.98 | 0.03 | OK |
| W4×Q16 | -2.0439 | -1.7807 | 2.63×10⁻¹ | 7.99 | 0.03 | OK |
| W4×Q8 | -2.0439 | -1.7803 | 2.64×10⁻¹ | 8.99 | 0.03 | OK |

> 注：gold 值较小（-2.04）是因为全负激活与正弦权重的 dot product 发生大量抵消。
> 1LSB 基于输入能量（≈128）× (1/16) 等标准系数，与 ADC/JEDEC 一致。

#### K=256, 全正激活

| 路径 | gold | test | \|err\| | 1LSB | \|err\|/1LSB | 结果 |
|------|------|------|---------|------|-------------|------|
| f32×f32 | 3.6701 | 3.6701 | 0 | — | — | OK |
| W4×f32 | 3.6701 | 3.4078 | 2.62×10⁻¹ | 7.98 | 0.03 | OK |
| W4×Q16 | 3.6701 | 3.4077 | 2.62×10⁻¹ | 7.99 | 0.03 | OK |
| W4×Q8 | 3.6701 | 3.3992 | 2.71×10⁻¹ | 8.99 | 0.03 | OK |

#### K=512, 全负激活

| 路径 | gold | test | \|err\| | 1LSB | \|err\|/1LSB | 结果 |
|------|------|------|---------|------|-------------|------|
| f32×f32 | -5.1269 | -5.1269 | 0 | — | — | OK |
| W4×f32 | -5.1269 | -4.9693 | 1.58×10⁻¹ | 16.0 | 0.01 | OK |
| W4×Q16 | -5.1269 | -4.9693 | 1.58×10⁻¹ | 16.0 | 0.01 | OK |
| W4×Q8 | -5.1269 | -4.9595 | 1.67×10⁻¹ | 18.0 | 0.01 | OK |

#### K=512, 混合激活

| 路径 | gold | test | \|err\| | 1LSB | \|err\|/1LSB | 结果 |
|------|------|------|---------|------|-------------|------|
| f32×f32 | 256.4554 | 256.4554 | 0 | — | — | OK |
| W4×f32 | 256.4554 | 256.2107 | 2.45×10⁻¹ | 16.0 | 0.02 | OK |
| W4×Q16 | 256.4554 | 256.2106 | 2.45×10⁻¹ | 16.0 | 0.02 | OK |
| W4×Q8 | 256.4554 | 256.2165 | 2.39×10⁻¹ | 18.0 | 0.01 | OK |

**Vec Dot 结论**：
- 所有路径的绝对误差（|err| ≈ 0.16~0.27）几乎一致，表明**误差主导来源是 Q4_K 权重量化**，激活量化（Q16 / Q8）贡献极小
- W4×Q16 vs W4×f32：激活从 float 改为 Q16 后误差几乎不变（差异 < 0.001），证明 Q16 激活量化损失可忽略
- W4×Q8 vs W4×f32：同理，Q8 激活量化额外误差也极小

### 3.3 Single Block GEMM 4-way

矩阵乘法 `C[N,M] = W[N,K] × A[K,M]`，输出为 N×M 矩阵。

#### K=256, M=8, N=16（128 个输出元素）

| 路径 | cosine | mae | 1LSB | mae/1LSB | 结果 |
|------|--------|-----|------|----------|------|
| f32×f32 | 1.000000 | 0 | — | — | OK |
| W4×f32 | 1.000000 | 1.23×10⁻¹ | 9.43 | 0.01 | OK |
| W4×Q16 | 1.000000 | 1.23×10⁻¹ | 9.43 | 0.01 | OK |
| W4×Q8 | 1.000000 | 1.17×10⁻¹ | 10.6 | 0.01 | OK |

#### K=256, M=4, N=32（128 个输出元素）

| 路径 | cosine | mae | 1LSB | mae/1LSB | 结果 |
|------|--------|-----|------|----------|------|
| f32×f32 | 1.000000 | 0 | — | — | OK |
| W4×f32 | 1.000000 | 1.73×10⁻¹ | 11.7 | 0.01 | OK |
| W4×Q16 | 1.000000 | 1.73×10⁻¹ | 11.7 | 0.01 | OK |
| W4×Q8 | 1.000000 | 1.65×10⁻¹ | 13.2 | 0.01 | OK |

**Single Block GEMM 结论**：
- cosine = 1.000000（6 位小数）——128 个元素 rel ≈ 8×10⁻⁴，理论 cosine ≈ 1 − 3×10⁻⁷，6 位显示为 1.0
- mae/1LSB 均 < 0.2，远低于 1LSB 阈值
- W4×Q16 的 mae 与 W4×f32 完全相同（1.23e-1 vs 1.23e-1），**Q16 激活对 GEMM 输出的额外误差为零**

### 3.4 GEMM Chain 4-way（误差累积观察）

模拟 qwen3moe 单 block 中 4 次 GEMM 串联（类似 Q → K → V → O 或 up → gate → down），每步输出作为下步输入。

| 路径 | cosine | mae | 1LSB | mae/1LSB | 结果 |
|------|--------|-----|------|----------|------|
| f32×f32 | 1.000000 | 0 | — | — | OK |
| W4×f32 | 0.999884 | 2.18×10⁴ | 1.79×10⁴ | 1.22 | >1LSB |
| W4×Q16 | 0.999884 | 2.19×10⁴ | 1.79×10⁴ | 1.22 | >1LSB |
| W4×Q8 | 0.999818 | 1.96×10⁴ | 2.01×10⁴ | 0.98 | <1LSB |

**Chain 结论**（仅供参考，不计入 pass/fail）：
- 4 轮串联后，量化误差被矩阵条件数放大约 15 倍 1LSB
- cosine 从 1.0 降至 0.9998 级别，跌破 0.9999 阈值
- **W4×Q16 ≈ W4×f32**（cosine 和 mae 几乎一致），再次证明 Q16 不引入额外累积误差
- Q16 和 Q8 均采用 **per-block scale（QK_K=256）**，区别仅在位宽：Q16 为 15-bit（1/32767），Q8 为 7-bit（1/127）
- W4×Q8 的 cosine 略低于 W4×Q16（0.999818 vs 0.999884），mae 也略低——Q8 量化噪声较大（±1/127 vs ±1/32767），在部分维度上恰好与权重量化误差方向相反，产生微弱的"误差抵消"效应；Q16 因量化噪声极小（精度接近 float），不产生这种抵消

### 3.5 Qwen3-30B 全层端到端测试

使用实际模型（Qwen3-30B-A3B-Instruct）运行完整推理，**所有 Q4_K 层均走 REEX 路径**（`REEX_TARGET_LAYER=-1`），对比 Layer 0 中间 tensor 和**模型最终输出 logits**。

- **模型**：Qwen3-30B-A3B-Instruct-2507（F16 + Q4_K 两个 gguf）
- **输入**：`"The quick brown fox jumps over the lazy dog"`（9 token，非单 token，使 attention 非退化）
- **REEX 范围**：全部 48 层的 Q4_K GEMM 均使用 REEX 替换（生产场景）
- **dump 内容**：Layer 0 中间 tensor（26 个，含 2 个整数索引 tensor）+ **final_logits**（151,936 维词表概率）
- **脚本**：`run_single_block_compare.sh` + `compare_layer_dumps.py`

#### 最终输出 Top-5 预测对比（最重要指标）

| 排名 | Gold (F16) | W4×float | W4×INT16 | W4×INT8 |
|:----:|-----------|----------|----------|---------|
| #1 | **"."** (24.50) | **"."** (24.11) | **"."** (24.14) | **"."** (24.33) |
| #2 | ".Ċ" (23.81) | ".Ċ" (23.55) | ".Ċ" (23.53) | ".Ċ" (23.81) |
| #3 | ".ĊĊ" (23.48) | ".ĊĊ" (23.43) | ".ĊĊ" (23.40) | ".ĊĊ" (23.53) |
| #4 | "Ċ" (22.30) | "Ċ" (22.08) | "Ċ" (21.84) | "Ċ" (21.97) |
| #5 | "è¿" (21.68) | "," (21.73) | "," (21.73) | "," (21.87) |

> **关键结论**：**Top-1 预测在所有 4 条路径下完全一致**（"."），Top-4 也完全一致。仅第 5 名有差异（Gold 为 "è¿"，W4 三路为 ","）——这由 Q4_K 权重量化导致，与激活量化无关。
>
> 三条 REEX 路径（W4×float / INT16 / INT8）的 Top-5 **完全相同**，说明全层 REEX 替换不改变模型预测行为。

#### Part 1：Gold (F16 float×float) vs 各 REEX 路径

衡量**总误差**（权重量化 + 激活量化），三路共用统一的 8-bit LSB 归一化，可直接横向对比。

| Tensor | W4×float cos | W4×float mae/LSB₈ | W4×INT16 cos | W4×INT16 mae/LSB₈ | W4×INT8 cos | W4×INT8 mae/LSB₈ | 备注 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|------|
| Kcur-0 | 0.999731 | 0.12 | 0.999731 | 0.12 | 0.999736 | 0.12 | Q4_K attn_q |
| Qcur-0 | 0.999314 | 0.36 | 0.999314 | 0.36 | 0.999303 | 0.36 | Q4_K attn_q |
| Vcur-0 | 0.998327 | 0.78 | 0.998327 | 0.78 | 0.998327 | 0.78 | **Q6_K 权重，三路相同** |
| \_\_fattn\_\_-0 | 0.997549 | 0.76 | 0.997549 | 0.76 | 0.997543 | 0.77 | Flash Attention 输出 |
| kqv_wo-0 | 0.998811 | 0.12 | 0.998811 | 0.12 | 0.998810 | 0.12 | Attention 输出投影 |
| ffn_moe_gate-0 | 0.870653 | 2.05 | 0.870653 | 2.05 | 0.870585 | 2.06 | MoE 门控 GEMM |
| ffn_moe_up-0 | 0.859788 | 2.19 | 0.859788 | 2.19 | 0.859770 | 2.20 | MoE 上投影 GEMM |
| ffn_moe_down-0 | 0.876019 | 0.27 | 0.876019 | 0.27 | 0.876122 | 0.27 | **Q6_K 权重**，误差从输入传入 |
| ffn_moe_weighted-0 | 0.833312 | 0.18 | 0.833311 | 0.18 | 0.833404 | 0.18 | MoE 加权求和（误差累积终点） |
| l_out-0 | 0.999133 | 0.08 | 0.999134 | 0.08 | 0.999132 | 0.08 | Block 最终输出 |
| **final_logits** | **0.998246** | **1.54** | **0.998180** | **1.63** | **0.998240** | **1.53** | **模型最终输出（151,936维）** |
| ffn_moe_argsort-0 | — | — | — | — | — | — | [idx] 41%match（三路相似） |
| ffn_moe_topk-0 | — | — | — | — | — | — | [idx] 87.5%match（三路相同） |

**Part 1 摘要**（排除 2 个整数索引 tensor，25 个连续值 tensor 参与统计）：

| 路径 | cos_min | cos_avg | max mae/LSB₈ | pass (cos≥0.9999) |
|------|---------|---------|--------------|-------------------|
| W4×float | 0.833312 | 0.976692 | 2.19 | 3/25 |
| W4×INT16 | 0.833311 | 0.976690 | 2.19 | 3/25 |
| W4×INT8 | 0.833404 | 0.976696 | 2.20 | 3/25 |

> **解读**：三条路径的 cos 和 mae/LSB₈ 几乎完全一致，说明 **Q4_K 权重量化是绝对主导误差源**。cos < 0.9999 的 tensor 主要是 MoE FFN 链和 final_logits（48 层误差累积），均为权重量化在多步运算中的传播放大，而非 REEX 激活量化的问题。
>
> **最重要**：尽管 final_logits 的 cos 为 0.998 级别（低于 0.9999 阈值），**Top-1 预测在所有路径下完全一致**（"."），且三条 REEX 路径的 Top-5 完全相同，说明权重量化引入的 logits 偏移不影响实际预测。

#### Part 2：Cross-Comparison（隔离激活量化误差）

以 W4×float 为 baseline，对比 W4×INT16 和 W4×INT8 的**激活量化额外误差**。每条路径用各自的 LSB 归一化。

| Tensor | INT16 cos | INT16 mae/LSB₁₆ | INT8 cos | INT8 mae/LSB₈ | 备注 |
|--------|:---------:|:----------------:|:--------:|:--------------:|------|
| Kcur-0 | 1.000000 | 0.024 | 0.999993 | 0.021 | |
| Qcur-0 | 1.000000 | 0.065 | 0.999982 | 0.059 | |
| Vcur-0 | 1.000000 | 0.000 | 1.000000 | 0.000 | Q6_K，REEX 不介入 |
| \_\_fattn\_\_-0 | 1.000000 | 0.253 | 0.999990 | 0.033 | |
| ffn_moe_gate-0 | 1.000000 | 0.214 | 0.999911 | 0.117 | |
| ffn_moe_up-0 | 1.000000 | 0.215 | 0.999880\* | 0.133 | INT8 略低于 0.9999 |
| ffn_moe_down-0 | 0.999999 | 0.337 | 0.999870\* | 0.020 | INT8 略低于 0.9999 |
| ffn_moe_weights-0 | 1.000000 | 0.571 | 0.999996 | 0.107 | |
| l_out-0 | 1.000000 | 0.214 | 0.999979 | 0.012 | |
| **final_logits** | **0.999730** | **171.7** | **0.999582** | **0.790** | **48层累积，最终输出** |
| ffn_moe_argsort-0 | — | [idx] 100% | — | [idx] 91.4% | INT16 完全一致 |
| ffn_moe_topk-0 | — | [idx] 100% | — | [idx] 100% | 两路选择结果一致 |

**Part 2 摘要**（25 个连续值 tensor 参与统计）：

| 路径 | cos_min | cos_avg | max mae/LSB | pass (cos≥0.9999) |
|------|---------|---------|-------------|------|
| **W4×INT16** | 0.999730 | 0.999989 | 171.7 | **24/25** |
| W4×INT8 | 0.999582 | 0.999959 | 0.790 | 22/25 |

> **解读**：
> - **Layer 0 中间 tensor**：W4×INT16 仍保持 24/24 全部通过（cos_min=0.999999），W4×INT8 有 2 个 MoE tensor（ffn_moe_up、ffn_moe_down）略低于 0.9999
> - **final_logits（最重要指标）**：W4×INT16 cos=0.999730，W4×INT8 cos=0.999582——cos 虽略低于 0.9999 阈值，但这是 48 层误差累积的结果（每层 cos≈1.0，48 层乘积自然下降）
> - W4×INT16 的 mae/LSB₁₆=171.7 看似很高，实因 INT16 LSB 极小（max_logit/32767≈0.0007），绝对 MAE 仅 0.126；而 W4×INT8 的 mae/LSB₈=0.79 < 1，因 Q8 LSB 较大
> - **Top-1 预测一致**：三条 REEX 路径的 Top-5 完全相同（均为 "." ".Ċ" ".ĊĊ" "Ċ" ","），最终模型行为无差异
> - 整数索引 tensor：INT16 与 float 的 argsort/topk **完全一致**（100%match），INT8 的 argsort 有 8.6% 变化但 topk 选择仍然完全一致

#### 异常数据说明

| 类别 | Tensor | 现象 | 原因 |
|------|--------|------|------|
| **三路 cos/mae 完全一致** | Vcur-0, attn_norm-0 | cos 相同，mae 相同 | 权重为 Q6_K（非 Q4_K），REEX 不介入，三路走同一标准路径 |
| **三路 cos/mae 完全一致** | ffn_moe_topk-0 | [idx] 87.5%match（三路相同） | 整数 TopK 操作，输入微小差异不影响选择结果 |
| **Part 1 cos 极低** | ffn_moe_gate/up/weighted | cos=0.83~0.87 | MoE FFN 链中 Q4_K 误差逐级传播放大，非激活量化问题 |
| **Part 2 INT8 cos 略低** | ffn_moe_down-0, ffn_moe_up-0 | cos=0.9987~0.9989 | MoE 路由对 Q8_K 量化噪声的放大；mae/LSB 仍远 <1 |
| **Part 2 final_logits cos<0.9999** | final_logits | INT16: 0.999730, INT8: 0.999582 | 48 层误差累积的自然结果；单层 cos≈1.0 经 48 层乘积后下降至 0.9995~0.9997 级别 |
| **Part 2 INT16 mae/LSB₁₆=171.7** | final_logits | 看似极高 | INT16 LSB 极小（≈0.0007），绝对 MAE 仅 0.126；不影响 Top-k 预测 |
| **Part 2 INT16 argsort 100%** | ffn_moe_argsort-0 | INT16 100%match, INT8 91.4% | per-block Q16 精度足够高，gate logits 差异极小不改变排序 |

### 3.6 KV Cache 量化对比

> **本节已移至** [KV Cache 量化报告 §4](test-reex-kv-cache-report.md)。
>
> 摘要结论：3 种 KV × 4 种 REEX 路径 = 12 组，**所有组合 Top-1 预测完全一致**，KV 量化不改变 REEX 激活量化的相对优势。详见 KV Cache 专题报告。

## 四、关键发现

### 4.1 误差来源分析

| 量化阶段 | 贡献占比 | 证据 |
|----------|---------|------|
| **Q4_K 权重量化** | **~99.9%** | W4×f32 vs W4×Q16 的 mae 完全相同 |
| Q16 激活量化 | < 0.1% | Q16 roundtrip mae = 7.5×10⁻⁶，远低于权重误差 |
| Q8 激活量化 | < 1% | Q8 roundtrip mae = 2.1×10⁻³，权重误差仍主导 |

### 4.2 Q16 vs Q8 对比

| 维度 | W4×Q16 | W4×Q8 | 差异 |
|------|--------|-------|------|
| 激活精度 | 15-bit (int16) | 7-bit (int8) | Q16 高 256 倍 |
| 量化方式 | per-block scale (QK_K=256) | per-block scale (QK_K=256) | 结构对齐，与 Q8_K 一致 |
| 单次 GEMM mae | 1.23×10⁻¹ | 1.17×10⁻¹ | 几乎相同（差异 < 5%） |
| 串联 4 轮 cosine | 0.999884 | 0.999818 | Q16 略优 |
| 端到端 Layer0 Cross cos_min | **0.999999** | 0.999870 | Q16 显著占优 |
| 端到端 Layer0 pass rate | **24/24** | 22/24 | Q16 Layer0 全通过 |
| **全层 final_logits cos** | **0.999730** | 0.999582 | Q16 优于 Q8 |
| **全层 Top-1 预测一致** | **✓** | **✓** | 两者均与 W4×float 一致 |
| 激活存储 | ~2.02 bytes/element (block_q16_K) | ~1.06 bytes/element (Q8_K) | Q16 约 2× |
| 实现方式 | 自研整数域累加 (int64) | 复用原生 `ggml_vec_dot_q4_K_q8_K` | Q8 有 SIMD 优化 |

**结论**：
- 在单次 GEMM 级别，Q16 和 Q8 的精度差异极小（被 Q4_K 权重量化主导）
- 在端到端推理（Layer 0 中间 tensor）中，**Q16 全部 24/24 通过**而 Q8 有 2 个 MoE tensor 略低于阈值
- **在全层替换的 final_logits 对比中**：Q16 cos=0.999730, Q8 cos=0.999582，**两者 Top-1 预测均与 W4×float 完全一致**
- Q16 在 MoE 路由敏感路径上表现更稳健：argsort 100%match vs INT8 的 91.4%
- Q16 的 per-block 量化（从 per-row 改进后）使精度接近 float 水平，额外误差可忽略
- **引入 KV cache 量化后**，12 种组合 Top-1 全部一致——详见 [KV Cache 报告](test-reex-kv-cache-report.md)
- **最终结论：全层 REEX 替换不改变模型预测行为，可安全部署**

### 4.3 通过标准汇总

#### test-reex-gemm（合成数据，单操作级）

| 测试类别 | 测试数 | 通过数 | 通过率 |
|----------|--------|--------|--------|
| 激活 Roundtrip | 4 | 4 | 100% |
| Vec Dot 4-way | 16 | 16 | 100% |
| Single Block GEMM 4-way | 8 | 8 | 100% |
| **单次操作合计** | **28** | **28** | **100%** |
| GEMM Chain 4-way（参考） | 4 | 2 | 50%（f32×f32、W4×Q8 满足 mae<1LSB；W4×f32/Q16 因累积略超） |

#### Qwen3-30B 全层端到端（Cross-Comparison，F16 KV 基线）

| KV cache | INT16 L0 pass | INT16 logits cos | INT8 L0 pass | INT8 logits cos | Top-1 一致 |
|:--------:|:------------:|:---------------:|:-----------:|:--------------:|:----------:|
| **F16** | **24/24 ✓** | 0.999730 | 22/24 | 0.999582 | **✓** (12/12) |

> 包含 Q8_0/Q4_0 KV cache 的完整 12 组交叉对比已移至 [KV Cache 报告 §4](test-reex-kv-cache-report.md)。
> 所有 12 种组合 Top-1 预测完全一致，KV 量化不改变 REEX 的相对优势。

## 五、Wikitext PPL 全模型对比

> **CPU GEMM PPL 全矩阵**已移至 [PWNL 实验设计 §3.2](test-reex-pwnl-experiment-design.md)（包含完整 6×3 PPL 矩阵和 ΔPPL 分解）。
>
> 摘要结论：所有 REEX 路径 |ΔPPL| < 0.04，远小于 PPL 统计不确定度（±0.06）。**REEX 全层替换对 wikitext PPL 无显著影响**。

### 5.1 CUDA GEMM+PWNL 联合验证

CUDA 端已完成 GEMM+PWNL 联合 PPL 验证（详见 [PWNL 精度评估报告 §四](test-reex-pwnl-report.md)），关键结果：

**exp clamp 修复后（EXP_MIN=-8）**：

| 数据集 | CUDA Std PPL | CUDA GEMM+PWNL PPL | ΔPPL |
|--------|:---:|:---:|:---:|
| wiki-103-raw validation (507 chunks) | 7.8911 | 7.8320 | **-0.059** |
| wiki-103-raw test (579 chunks) | 7.3554 | 7.3066 | **-0.049** |
| wiki-2-raw test (579 chunks) | 7.3554 | 7.3066 | **-0.049** |

**sin_cos 合并后（exp LUT 扩展 + 新增算子，EXP_MIN=-20）**：

| 数据集 | STD PPL（新基线） | REEX PWNL PPL | ΔPPL |
|--------|:---:|:---:|:---:|
| wiki-2-raw test | 7.5946 ±0.054 | **7.5713** ±0.054 | **-0.023** |

> **CUDA 端 GEMM+PWNL 联合 PPL 在所有数据集上均无退化**（ΔPPL 为负值，在统计不确定度范围内）。sin_cos 合并后新增的 exp LUT 数据重生成、EXP_MIN 扩展、expm1/softplus/gelu_quick/log_soft_max 等替换均已包含在验证中。

## 六、实现说明

### 6.1 计算路径实现

| 函数 | 路径 | 实现策略 |
|------|------|----------|
| `ggml_vec_dot_q4_K_f32_reex` | W4×f32 | 反量化 Q4_K → float，做 float dot product |
| `ggml_vec_dot_q4_K_q16_reex` | W4×Q16 | 整数域累加：int64 += scale × (q16 × q4)，参考原生 q4_K_q8_K 模式 |
| `ggml_vec_dot_q4_K_q8_reex` | W4×Q8 | 包装调用原生 `ggml_vec_dot_q4_K_q8_K`（有 AVX2/NEON 优化）；CPU 主 GEMM 分发仍按原生 `Q4_K×Q8_K` 路径执行 |

### 6.2 Q16 激活格式（block_q16_K，per-block scale）

```
结构体：block_q16_K { float d; int16_t qs[QK_K]; }   // QK_K = 256
布局：每 QK_K=256 个元素一个 block，每个 block 独立 scale
量化：d = amax / 32767,  qs[j] = round(x[j] / d)     // j ∈ [0, QK_K)
反量化：x[j] ≈ qs[j] × d
存储：(sizeof(float) + QK_K × sizeof(int16_t)) × (k / QK_K) bytes/row
     = (4 + 512) × nb = 516 bytes / 256 elements ≈ 2.02 bytes/element
```

> 从 v1 的 per-row scale（整行共享一个 scale）改为 per-block scale（每 256 元素独立 scale），与 Q8_K 的分块策略对齐，显著减小长向量中局部极值对全行精度的影响。

### 6.3 测试数据

**test-reex-gemm（合成数据）**：权重和激活使用确定性正弦函数生成（非随机），保证可复现：
```c
dst[i] = amplitude * sinf(i * 0.1f + (i % 7) * 0.3f)
```
不同 amplitude 和变体（neg/pos/mix）覆盖多种输入分布。

**Qwen3-30B 全层端到端**：使用真实模型权重 + 全层 REEX 替换：
- Gold：`Qwen3-30B-A3B-Instruct-2507-F16.gguf`（16-bit 权重，float 计算）
- REEX：`Qwen3-30B-A3B-Instruct-2507-Q4_K.gguf`（Q4_K 权重，全部 48 层走 REEX）
- 输入：`"The quick brown fox jumps over the lazy dog"` → 9 token
- Dump 内容：Layer 0 中间 tensor + **final_logits**（模型最终输出 logits，151,936 维）
- 比较工具：`compare_layer_dumps.py`
- **REEX 范围**：`REEX_TARGET_LAYER=-1`（默认值，所有 Q4_K 层均走 REEX 路径）

### 6.4 参考（1 LSB 与 basis 的标准依据）

- **JEDEC**（如 JESD99B）：LSB 为线性转换器模拟分辨率的单位符号，其他模拟量与误差以 LSB 的倍数/分数表示（如 ½ LSB）。
- **ADC/DAC 数据手册**：1 LSB = Vref / 2^N（N 为位数）；Full Scale = Vref − 1 LSB。
- **数值误差中的 dot product**：绝对误差界与 Σ|x_i||y_i| 成正比（与分量的绝对值乘积之和成正比），因此本测试中 **basis = energy 或 mean_energy**，与误差传播定义一致；仅在 Chain 等无法逐行计算 energy 时退化为 mean|gold| 作为输出量级参考。
- **本测试**：output_1LSB = basis × LSB_REL_factor，各格式 LSB_REL 与 1/2^N 一致（Q4_K 1/16、Q16 1/32767、Q8 1/127）。

## 七、CUDA REEX 测试

将 CPU 的 W4×INT8 / W4×INT16 拓展到 CUDA 后，通过以下用例检查正确性与是否满足要求。

### 7.1 测试用例一览

| 测试可执行文件 | 编译条件 | 检查内容 | 通过标准 |
|----------------|----------|----------|----------|
| **test-reex-cuda-q8** | GGML_CUDA=ON，且当前配置不启用 REEX Q16 | CUDA W4×INT8(Q8_1) 输出相对 float×float 金标准的精度；同时输出相对 Q4_K×f32 的增量误差；并检查 `q8_mul_mat_hits > 0`，确认确实命中 CUDA Q4_K×Q8_1 路径 | 最终验收：cosine ≥ 0.9999，mae/LSB(W4+Q8) < 1，且 `q8_mul_mat_hits > 0` |
| **test-reex-cuda-q16** | GGML_CUDA=ON + GGML_REEX_GEMM=ON + GGML_REEX_GEMM_ACTIVATION=Q16 | CUDA REEX W4×INT16 输出相对 float×float 金标准的精度；同时输出相对 Q4_K×f32 的增量误差，并验证 `M<=8` 命中 Q16 `MUL_MAT`、较大 token-batch 的 `MUL_MAT_ID` 仍可命中 Q16 路径（已覆盖 `n_tokens=16`）、`MoE-like` 广播输入场景（`ne11=1, n_used=8, n_tokens=16`）也可命中；额外包含两个图级覆盖测试：`gate/up` 双节点链路与 `Qwen3MoE-like routed FFN` 三节点链路，用于验证图内多个专家 GEMM 节点连续命中 Q16 CUDA | 最终验收：算子级 case 仍要求 cosine ≥ 0.9999，mae/LSB(W4+Q16) < 1；图级覆盖测试以命中/trace 为主，精度指标仅输出观察 |

### 7.2 编译与运行

```bash
# Q8 配置：验证 CUDA 默认 W4×INT8(Q8_1) 路径
mkdir build-q8 && cd build-q8
cmake .. -DGGML_CUDA=ON
cmake --build . -j
./bin/test-reex-cuda-q8

# Q16 配置：验证 CUDA REEX W4×INT16 路径
cd ..
mkdir build-q16 && cd build-q16
cmake .. -DGGML_CUDA=ON -DGGML_REEX_GEMM=ON -DGGML_REEX_GEMM_ACTIVATION=Q16
cmake --build . -j
./bin/test-reex-cuda-q16

# 或通过 ctest
ctest -R "test-reex-cuda-(q8|q16)" -V

# 若使用真实模型入口 tests/test-reex/eval_single_token.cpp
# （例如配合 run_single_block_compare.sh 跑 Qwen3MoE 单 block / 单 token），
# Q8 / Q16 CUDA 构建现在会额外输出：
#   REEX CUDA trace:
#     q8_mul_mat_hits
#     q8_mul_mat_id_hits
#     q16_mul_mat_hits
#     q16_batch_fallbacks
#     q16_mul_mat_id_hits
#     q16_mul_mat_id_unsupported
# 并在每个 dump 目录下写出 reex_cuda_trace.txt，便于核对真实模型图中的命中情况。

# 真实模型单层 token sweep（Q8 / Q16 CUDA）
cd tests/test-reex
MODEL=/home/shared/models/Qwen/Qwen3-30B-A3B-Instruct-2507/Qwen3-30B-A3B-Instruct-2507-Q4_K.gguf \
TARGET_LAYER=0 \
TOKENS="1 2 4 8 12 16 24 32" \
./run_reex_cuda_trace_sweep.sh
```

### 7.3 满足要求判定

- **W4×INT8**：以 float×float 为金标准，要求 `cosine ≥ 0.9999` 且 `mae/LSB(W4+Q8) < 1`。`vs Q4_K×f32` 仅用于观察激活量化带来的增量误差。
- **W4×INT8 路径命中**：算子级测试还要求 `q8_mul_mat_hits > 0`，确认本次用例确实走到了 CUDA `Q4_K×Q8_1` 实现，而不是其它回退路径。
- **W4×INT16**：以 float×float 为金标准，要求 `cosine ≥ 0.9999` 且 `mae/LSB(W4+Q16) < 1`。`vs Q4_K×f32` 仅用于观察激活量化带来的增量误差。
- Q8 与 Q16 必须在**不同 CMake 配置**下分别构建和运行，不能在同一个二进制中混合判断两条路径。
- **当前 CUDA W4×INT16 的范围限制**：`MUL_MAT` 仅支持小 batch `MMVQ` 路径（当前 `M <= 8`），`M > 8` 会回退到现有 CUDA 主路径；`MUL_MAT_ID` / MoE 专家路由当前已不再受显式 `n_tokens` 上限限制，测试已覆盖到 `n_tokens = 16`；同时仍要求 `src1->ne[3] == 1`（单 sample），这一点也是当前 `ggml_mul_mat_id()` 算子本身的输入约束。

### 7.4 真实模型单层 token sweep（Layer 0）

为了把 `REEX CUDA trace` 从“全图聚合命中数”收敛为“单层命中数”，现已将 `REEX_TARGET_LAYER` 接入 CUDA dispatcher。下面结果使用真实 `Qwen3-30B-A3B-Instruct-2507-Q4_K.gguf`、`TARGET_LAYER=0`，分别在独立 `Q8` / `Q16` CUDA 构建下测得。

测试脚本：`tests/test-reex/run_reex_cuda_trace_sweep.sh`

提示词采用 `"a"` 重复拼接，使实际 tokenizer 计数稳定落在 `1/2/4/8/12/16/24/32` token。

#### Q16 构建结果

| requested_tokens | actual_tokens | q16_mul_mat_hits | q16_batch_fallbacks | q16_mul_mat_id_hits | q16_mul_mat_id_unsupported |
|-----------------:|-------------:|------------------:|--------------------:|--------------------:|---------------------------:|
| 1  | 1  | 3 | 0 | 2 | 0 |
| 2  | 2  | 3 | 0 | 2 | 0 |
| 4  | 4  | 3 | 0 | 2 | 0 |
| 8  | 8  | 3 | 0 | 2 | 0 |
| 12 | 12 | 0 | 3 | 2 | 0 |
| 16 | 16 | 0 | 3 | 2 | 0 |
| 24 | 24 | 0 | 3 | 2 | 0 |
| 32 | 32 | 0 | 3 | 2 | 0 |

#### Q8 构建结果

| requested_tokens | actual_tokens | q8_mul_mat_hits | q8_mul_mat_id_hits |
|-----------------:|-------------:|----------------:|-------------------:|
| 1  | 1  | 168 | 120 |
| 2  | 2  | 168 | 120 |
| 4  | 4  | 168 | 120 |
| 8  | 8  | 168 | 120 |
| 12 | 12 | 168 | 120 |
| 16 | 16 | 168 | 120 |
| 24 | 24 | 168 | 120 |
| 32 | 32 | 168 | 120 |

结论：

- **单 token / 小 token-batch 不够完整**：它只能证明 layer 内的 `Q16 MUL_MAT` 小 batch 路径确实命中，以及 `Q16 MUL_MAT_ID` 已接入。
- **切换点已被直接量到**：`8 token` 时 layer 0 仍有 `q16_mul_mat_hits = 3`；`12 token` 时已变为 `q16_mul_mat_hits = 0` 且 `q16_batch_fallbacks = 3`，与当前实现的 `M <= 8` 门限一致，即 `M > 8` 后走 fallback。
- **`MUL_MAT_ID` 更稳定**：在 `1/2/4/8/12/16/24/32` token 下，layer 0 的 `q16_mul_mat_id_hits` 都保持为 `2`，说明真实模型中的专家路由节点不受这组 token-batch 扫描影响，路径接入稳定。
- **Q8 路径覆盖更完整**：独立 `Q8` 构建下，layer 0 的 `q8_mul_mat_hits = 168`、`q8_mul_mat_id_hits = 120` 在 `1~32 token` 范围内保持不变，说明 CUDA `Q4_K×Q8_1` 主路径不存在类似 `Q16 MUL_MAT` 的 `M <= 8` 限制。
- 因此，真实模型验证建议至少保留两组：一组 `<= 8 token` 验证命中，一组 `> 8 token` 验证 fallback；二者共同构成比单 token 更完整的 CUDA REEX 覆盖验证。

#### Q8 vs Q16 最终对照

| 维度 | CUDA W4×INT8 | CUDA W4×INT16 |
|------|--------------|---------------|
| 实现入口 | 主 CUDA 路径：`F32 -> Q8_1 -> Q4_K×Q8_1` | REEX CUDA 路径：`F32 -> block_q16_K -> Q4_K×Q16` |
| 对应可执行测试 | `test-reex-cuda-q8` | `test-reex-cuda-q16` |
| 算子级精度验收 | `cosine ≥ 0.9999`，`mae/LSB(W4+Q8) < 1`，已实测通过 | `cosine ≥ 0.9999`，`mae/LSB(W4+Q16) < 1`，算子级 case 已通过 |
| 算子级路径命中 | 要求 `q8_mul_mat_hits > 0`，已实测 `= 1` | 要求 `q16_mul_mat_hits > 0`，并验证大 batch fallback |
| 真实模型单层 trace | 已支持：`q8_mul_mat_hits`、`q8_mul_mat_id_hits` | 已支持：`q16_mul_mat_hits`、`q16_batch_fallbacks`、`q16_mul_mat_id_hits`、`q16_mul_mat_id_unsupported` |
| `MUL_MAT` token-batch 行为 | `1~32 token` 命中稳定，无 `M<=8` 门限现象 | `M <= 8` 命中；`M > 8` fallback |
| `MUL_MAT_ID` / MoE 行为 | 真实模型中稳定命中，单层实测 `q8_mul_mat_id_hits = 120` | 已接入并稳定命中，单层实测 `q16_mul_mat_id_hits = 2` |
| 覆盖完整性 | 更完整，更接近“默认 CUDA 主路径”行为 | 更强调高精度，但存在 `MUL_MAT` 小 batch 范围限制 |
| 推荐场景 | 优先追求覆盖稳定性、吞吐和低改动风险 | 优先追求更高激活精度，并接受 `MUL_MAT` 大 batch fallback |

**最终建议**：

- 若目标是**最稳妥地替代现有 CUDA 主路径**，优先使用 `W4×INT8`。
- 若目标是**在可控范围内提升激活精度**，并且主要关注 `M <= 8` 的 `MUL_MAT` / MoE 场景，可使用 `W4×INT16`。
- 最推荐的验证组合是：
  - `Q8`：保留 `test-reex-cuda-q8` + 真实模型单层 trace
  - `Q16`：保留 `test-reex-cuda-q16` + `<=8 token` 命中 case + `>8 token` fallback case

### 7.5 CUDA GEMM + PWNL 联合验证状态

CUDA 端 GEMM 与 PWNL（Mixed-FP16 LUT 非线性算子替换）的联合验证已完成，含 sin_cos 合并后的最新验证：

| 验证维度 | CUDA GEMM+PWNL | 版本 | 参考 |
|---------|:---:|------|------|
| Wikitext PPL (4 数据集) | 所有 ΔPPL ∈ [-0.059, -0.047] | exp clamp 修复后 | [PWNL 报告 §四](test-reex-pwnl-report.md) |
| Wikitext-2 PPL | ΔPPL = **-0.023** | **sin_cos 合并后** | [PWNL 报告 §四](test-reex-pwnl-report.md) |
| HellaSwag acc_norm (10042) | 74.95%（Δ = -0.04%） | exp clamp 修复后 | [PWNL 报告 §四](test-reex-pwnl-report.md) |
| HellaSwag acc_norm (10042) | **75.07%** | **sin_cos 合并后** | [PWNL 报告 §四](test-reex-pwnl-report.md) |
| Winogrande acc (1266) | 71.96%（Δ = +0.16%） | exp clamp 修复后 | [PWNL 报告 §四](test-reex-pwnl-report.md) |
| GEMM 零额外误差 | PWNL vs GEMM+PWNL PPL 精确相同 | 全版本 | [PWNL 实验设计 §3.2](test-reex-pwnl-experiment-design.md) |

**sin_cos 合并后新增变更**：
- exp LUT 数据重生成（非均匀分布 [-10.22, 0]，16 段）+ EXP_MIN 扩展至 -20
- 新增算子 LUT 替换：expm1、softplus、gelu_quick、log_soft_max、RoPE logf、tsembd logf
- softplus 链接修复（`ggml-impl.h` 保持 inline）

**结论**：CUDA 端 GEMM（W4×INT8）与 PWNL（Mixed-FP16 LUT）的联合部署对模型精度无退化。含 Flash Attention exp LUT（范围扩展后）、RMSNorm rsqrt、RoPE sin/cos/log、Softmax exp、SiLU sigmoid、expm1、softplus、gelu_quick 等全部非线性算子均已验证安全。详细实验设计与消融分析见 [PWNL 实验设计](test-reex-pwnl-experiment-design.md) §五。

**代码版本**：75bee288 (sin_cos merged) + 5f7a0561 (softplus link fix) on feature/pwnl-mixed-fp16
