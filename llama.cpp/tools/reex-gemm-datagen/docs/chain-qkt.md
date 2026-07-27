# 两级串联 MMA 采数（`--chain qkt`）— 设计说明

在现有单次量化 GEMM（`C = A @ Wᵀ`）基础上，新增一条**两次连续 MMA** 的采数链路，
对应注意力里"先算出 K、再算 QKᵀ"的融合计算。核心价值：**中间结果不反量化、不重新量化**——
第一个 MMA 的输出级直接吐出量化 block，第二个 MMA 原样读取，量化只做一遍。

> 背景与单次 GEMM 的量化/tiling/落盘约定见 [`DESIGN.md`](DESIGN.md)；本文只讲这条 chain 的增量部分。

---

## 1. 需求与语义（图片对应）

```
1st mma: [1,64][64,64]->[1,64] -> scale@FP16 + 64*element@INT4/INT8
  · 连续 64 行(64 组)凑够 Rslt[64,64]
2nd mma: QK^T, K 即为 1st mma Rslt
         [1,64][64,64]->[1,64]
```

用注意力语义命名三个维度（均为 64）：

| 记号 | 含义 | 在两级里的角色 |
|---|---|---|
| `head_dim` | 头维度 = 64 | MMA1 输出宽度 `N1` = MMA2 收缩宽度 `K2` |
| `num_keys` | key/token 数 = 64 | MMA1 行数 `M1` = MMA2 右操作数行数 `N2` |
| `num_queries` | query 数 = 1 | MMA2 行数 `M2` |

- **MMA1**：`Rslt[m1,n1] = Σ_{k1} A1[m1,k1]·W1[n1,k1]`，`m1=key`、`n1=head_dim`。造出 K 矩阵 `K[key, head_dim]`。
- **MMA2**：`S = Q @ Kᵀ`，`S[m2,n2] = Σ_{k2} Q[m2,k2]·K[n2,k2]`，`m2=query`、`n2=key`、`k2=head_dim`。

### 1.1 为什么"省下量化"、且"不用逻辑转置"

- 量化分组永远沿**收缩轴**做（scale 才能提到求和外）。MMA1 每算一行输出 `[1,64]`，就把这 64 个
  `head_dim` 值当**一个量化组**（一个 fp16 scale + 64 个 INTx 码）。
- 公式里的 `Kᵀ` 让 `head_dim` 从 MMA1 的**输出轴**变成 MMA2 的**收缩轴**。于是 MMA1 顺手按输出行做的
  量化，**恰好**就是 MMA2 想要的 K 方向量化 → 直接拿来用。
- 因此：**不反量化、不重量化**；中间那份 `{fp16 d; INTx qs[64]}` 一鱼两吃——既是 MMA1 的输出，
  又是 MMA2 的 weight 输入（同一字节格式，就是现有 `block_q8_0_64` / `block_q4_0_64`）。
- 若硬件有独立的"块转置"单元（本工具建模之，见 §4），它是**以整块为单位**搬 `key↔head_dim` 的块网格，
  **组内 64 码不拆**，所以仍不触发反量化/重量化。

---

## 2. 形状与 tiling（决策 Q1/Q2）

| 级 | 工具约定 | 形状 | tiling |
|---|---|---|---|
| MMA1 | `A1[M1,K1] @ W1[N1,K1]ᵀ → Rslt[M1,N1]` | `[64,64] @ [64,64] → [64,64]` | Legacy 默认 `{Mt=4, Nt=64, Kt=64, wgroup=64, agroup=64}` |
| MMA2 | `Q[M2,K2] @ K[N2,K2]ᵀ → S[M2,N2]` | `[1,64] @ [64,64] → [1,64]` | **专用** `{Mt=1, Nt=64, Kt=64, wgroup=64, agroup=64}` |

- MMA2 的 `M2=1` 用 `Mt=1` 专用 tiling 直接满足，无需补齐；槽位公式对 `Mt` 通用。
- `head_dim=64 == Kt`，每行输出恰好 1 个量化组；`num_keys=64 == Nt`，`Ntiles=1`。

---

## 3. 量化配置（决策 Q3/Q5/Q8/Q9/Q10）

| 项 | 配置 | 默认 | 算法 |
|---|---|---|---|
| MMA1 激活 A1（X） | stage-1 在片量化，`--abits` | A8 | `act_quant`：`amax→scale(fp16)→RNE→对称补码 clamp`，`qmax=2^(b-1)-1` |
| MMA1 权重 W1（Wk） | 离线量化，`--wtype` | `q8_0_64`（W8） | 现有 `wt.encode` + `reorder` |
| **MMA1 融合输出 K** | 新输出级，`--kbits 8\|4` | INT8 | 与 `act_quant` 同一套，但**分组轴 = 输出行**；源为 **fp32 累加器** |
| MMA2 激活 Q | 随机 fp16 + stage-1，`--abits` | A8 | 同 A1（单一 `--abits`，两级共用） |
| MMA2 权重 K | = MMA1 融合输出（`q8_0_64`/`q4_0_64`） | 随 `--kbits` | 不再量化，直接读 |
| psum 截断 | `--psum`，两级同值 | 0（不截断） | — |

- **K_blocks 字节格式**：`INT8 → block_q8_0_64 {fp16 d; int8 qs[64]}`（66B）；
  `INT4 → block_q4_0_64 {fp16 d; nibble qs[64]}`（34B，nibble 交错：`e<32`→低半字节，`e≥32`→高半字节）。对称、无 `s` 求和字段。
- **融合量化源精度**：直接取 MMA1 的 **fp32 累加器**（避免"先 fp16 再 int"的双重舍入）。

---

## 4. 中间转置建模（决策 Q6/Q7）

硬件含一个独立的**块转置单元**，本工具显式建模并落盘"转置前/后"两份以驱动其验证。

- **转置前 `kblocks_pretrans`**：MMA1 输出级按 token 顺序落块（`key`-major），即
  `[num_keys 行 × head_dim]` 的行分组块布局。
- **块转置**：以整块为单位交换 `key ↔ head_dim` 的**块网格**（组内 64 码保持不动）。
- **转置后 `kblocks_posttrans`**：重排成 MMA2 读 weight 的 `weight_block_slot(n2, sb, N2, ts_mma2)` 顺序，
  即 `mma2_weight_blocks`。

> 在全 64 形状下，块网格为 `[64 × 1]`，转置前/后线性重合（都是 key 线性序 0..63）；
> 该机制对 `head_dim > 64`（多组）时会产生真实重排。两份仍都落盘。

---

## 5. 数据流

```
X[64,64] ──stage1量化──→ mma1_act_blocks ┐
                                          ├─ MMA1 (现有 GEMM, Mt=4) ─→ Rslt fp32 (result 序)
Wk[64,64] ─离线量化───→ mma1_weight_blocks ┘                              │
                                                                         ▼  ★输出级 kernel (device, 一线程一输出行)
                                                             每行[1,64] → amax → scale(fp16) → INTx 码
                                                                         │
                                                              kblocks_pretrans (token 序)
                                                                         │  ★整块转置 (key↔head_dim)
                                                                         ▼
                                              kblocks_posttrans  =  mma2_weight_blocks (weight 序)
Q[1,64] ──stage1量化──→ mma2_act_blocks ─────────────────────────────────┤
                                                                         ▼
                                              MMA2 (现有 GEMM, Mt=1) ─→ scores[1,64] → 11 dtype + golden
```

---

## 6. 产物清单（决策 Q11/Q12）

入口：`--chain qkt`；目录：`<out>/qkt-A{a}W{w}-K{k}-...-YYYYMMDD-HHMMSS/`。

| 文件 | 含义 | 顺序 |
|---|---|---|
| `mma1_act_src_<DT>.bin` | X 源（量化前，native dtype） | §4.1 act tile 序（MMA1 tiling） |
| `mma1_weight_src_f16.bin` | Wk 源（fp16） | §4.1 weight tile 序 |
| `mma1_act_blocks.bin` | A1 量化块 `{f16 d; intX qs[64]}` | 块-tile 序 |
| `mma1_weight_blocks.bin` | W1 量化块 | 块-tile 序 |
| `mma1_rslt_f32.bin` | MMA1 fp32 golden（融合量化前实值） | result tile 序（MMA1 tiling） |
| `kblocks_pretrans.bin` | ★MMA1 融合量化输出（转置前，token 序） | token 序 |
| `kblocks_posttrans.bin` | ★= `mma2_weight_blocks`（转置后，weight 序） | weight_block_slot（MMA2 tiling） |
| `mma2_act_src_<DT>.bin` | Q 源（fp16） | §4.1 act tile 序（MMA2 tiling） |
| `mma2_act_blocks.bin` | Q 量化块 | 块-tile 序 |
| `mma2_output_<DT>.bin` | scores（11 份，无 scale） | result tile 序（MMA2 tiling） |
| `mma2_golden_f32.bin` | MMA2 fp32 golden | 同 output |
| `meta.json` | 自描述：两级形状/tiling/量化/转置映射/verify | — |

> MMA1 **不落** 11 dtype cast 输出（链路不用）；MMA2 落全套。

---

## 7. 校验（决策 Q10）

链一致原则：**GPU 产出的 K_blocks 为唯一真值**，同时喂给 GPU-MMA2 与 CPU-golden-MMA2，
保证两路输入完全一致（沿用"以 GPU 为准、golden 用 double 仅验正确性 ~1e-5"的一贯做法）。

1. **MMA1**：GPU fp32 acc vs CPU 整数 golden（`golden_cpu_symmetric`）逐元素比对；`psum=0` 时附 dequant 交叉校验。
2. **K_blocks 逐比特交叉校验**：用**同一份 GPU fp32 累加器**在 CPU 复算融合量化，结果应与 GPU 输出级 kernel **逐比特相同**（验证输出级实现正确）。
3. **MMA2**：GPU vs CPU golden（两路共用 GPU K_blocks）逐元素比对。

`meta.json` 的 `verify` 段记录三处的 `max_abs/max_rel/mse` 及 K_blocks 比特一致性。

### 7.1 独立复算脚本 `tests/check_chain_qkt.py`

NumPy 独立复算整条链（不参与 C++ 构建）：
- de-tile 两级源数据；按 §3 算法复算 A1/W1/Q 量化；
- 复算 MMA1 整数 MAC → fp32 → 融合量化 → 与 `kblocks_pretrans` 比对；
- 复算块转置 → 与 `kblocks_posttrans` 比对；
- 复算 MMA2 → 与 `mma2_golden_f32` / `mma2_output_*` 比对。

---

## 8. 实现改动点（模块，已落地）

| 文件 | 改动 |
|---|---|
| 新增 `src/outquant.{cuh,cu}` | **输出级 device kernel**（读 MMA1 fp32 结果 → 每输出行分组量化 → 写 `block_q8_0_64`/`q4_0_64`，pretrans 序）+ host 包装 `out_quant_run_host` + CPU 复算 `out_quant_cpu`（同一 `RGD_HD` 例程，构造性逐比特一致）+ 整块网格转置 `kblocks_grid_transpose` |
| `src/datagen.{h,cpp}` | 新增 `datagen_fill_act`（MMA2 的 Q 向量生成，种子去相关） |
| `src/dumper.{h,cpp}` | 新增 `dump_chain_case` + `ChainVerify` + chain 版 `meta.json` |
| `src/main.cpp` | `--chain qkt` / `--kbits 8\|4` 解析；`run_chain_qkt` 串联 MMA1 → 输出级 → 转置 → MMA2（MMA2 tiling `{1,64,64,64,64}` 就地构造，未改共享库） |
| `CMakeLists.txt` | 加入 `src/outquant.cu` |
| `tests/check_chain_qkt.py` | 独立复算校验（新增） |
| `README.md` | 链路用法与索引 |

---

## 9. CLI

```
reex-gemm-datagen --chain qkt [--wtype q8_0_64|q4_0_64] [--abits 16|8|4]
                  [--kbits 8|4] [--actin ...] [--psum B] [--seed S] [--out DIR]
# 形状固定：MMA1 [64,64,64] / MMA2 [1,64,64]（head_dim=64）
```

示例：

```bash
reex-gemm-datagen --chain qkt --wtype q8_0_64 --abits 8 --kbits 8   # A8×W8, K=INT8
reex-gemm-datagen --chain qkt --wtype q4_0_64 --abits 8 --kbits 4   # A8×W4, K=INT4
```
