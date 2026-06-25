# REEX GEMM datagen — 通用 kernel 重构 Plan(在片量化 + 黑盒接口)

> 本文件是定稿设计与执行计划,供跨上下文续作。所有决策均已与用户确认(见末尾「决策记录」)。

## 0. 目标

把数据采集工具改成**镜像硬件三级流水**、且 kernel **按类型黑盒化**,以便一套代码对接不同
(权重类型 × A_bits × act_in × 输出类型)组合;采数数据喂给模拟器(输入 = `act_src` + `weight_blocks`),
模拟器输出与 GPU 采数输出做比对。CPU golden 仅验正确性(double,~1e-5,**不要求逐 bit**)。

固定形状:A[8192,2048] @ W[2048,2048]^T = C[8192,2048],`C = A @ Wᵀ`。

## 1. 流水线(芯片三级,GPU 采数镜像)

```
stage-0 离线(host):  权重 → weight_blocks   reex 参考量化器(roundf,半数远离零)
stage-1 在片量化单元(device kernel): act_src(fp 原生 act_in) → act_blocks{fp16 d; intX q}
        每 agroup 一组: amax(fp32) → scale = amax/qmax(存 fp16) → 半数向偶数 → clamp(±qmax)
        qmax = 2^(A_bits-1)-1 ; 容器 A16=int16, A8/A4=int8 ; 激活不带 sum
stage-2 通用 GEMM(device kernel): act_blocks + weight_blocks
        块内 int32 MAC →(--psum 可截断)→ fp32 折算 + fp32 块间累加
        → OutConv(在片) → 一次产出全部 6 种 dtype
```

- **stage-1 是 device kernel**(每行/每组量化一次,act_blocks 拷回落盘,作为芯片中间产物;不再是输入)。
- CPU golden 读同一份 act_blocks(double)做正确性比对。
- IntBlock 纯整数路径**独立**(裸 int、tiling[16,16]、int32 精确、整数 clamp),**仅共享 OutConv**。

## 2. 通用 GEMM kernel 的 4 个黑盒(全部 `RGD_HD`,GPU/CPU golden 共用)

| 黑盒 | 接口职责 | 类型轴 |
|---|---|---|
| `ActReader`   | 读 `{fp16 d; intX q}`,按 A_bits 取元素(int16/int8)→ int;给出组 scale | A16/A8/A4 |
| `WeightCodec` | 解码 block → 权重码 + scale(s);提供 `K_per_block`、`block_t` | q8_0_64 / q8_1_64s / q4_0_64 / Q5_K_64S / Q6_K_64 |
| `Dot`         | 块内 int32 MAC + `psum_trunc` + fp32 折算(= WeightCodec 解码 + ActReader 读 + MAC) | 随 WeightCodec |
| `OutConv`     | fp32 acc → 字节;float=RNE cast,int=半数向偶数 + 两补码饱和 `[-(qmax+1),qmax]` | F16/BF16/I16/I8/I6/I4 |

- 现有 `qmac.cuh` 的 `rgd_*_dot_block/superblock` 收敛为各 `WeightCodec::dot`(已是 `RGD_HD`)。
- 运行期 dispatch:kernel 内 `switch(wtype)`、`switch(A_bits)`(线程一致分支,不发散)。
- 一个大 kernel:`for kb: acc += Dot(...); OutConv::write_all(outs, idx, acc)`(遍历 6 种 dtype)。

## 3. 关键定值(已确认)

- 激活:**不带 sum**;容器 A16=int16 / A8·A4=int8;量化舍入 = **半数向偶数**;scale = **fp16**。
- 权重:**离线**量化(roundf,reex 参考器);W8 = `q8_0_64`/`q8_1_64s`、W4 = `q4_0_64`。
- 输出:6 种一次产全;INT = **半数向偶数** + 两补码 `[-(qmax+1),qmax]`;F16/BF16 = RNE。
- 中间精度:块内 INT32、块间 FP32;**不要求**与 CPU 逐 bit(golden 用 double 验正确性)。
- IntBlock:独立 kernel,仅共享 OutConv。
- act_in:`F32/F16/BF16/E5M2/E4M3`(device 需要 fp8 解码,移植 `fp8.h` 的 decode 为 `RGD_HD`)。
- 模拟器输入 = `act_src_<DT>.bin` + `weight_blocks.bin`;`act_blocks.bin` = 芯片中间产物(stage-1 产出)。

## 4. 改动文件清单与职责

| 文件 | 改动 |
|---|---|
| `convert.cuh`(新) | `RGD_HD` OutConv:`rne_lround`(下沉)、饱和、fp16/bf16 cast、`write_all(outs, idx, accf)` 与 `write_all_int(outs, idx, acc_i32)` |
| `fp8.h` | decode 改 `RGD_HD`(供 device 解码 act_src);encode 仍 host |
| `actquant.cuh/.cu`(新) | stage-1 device 量化 kernel:`act_src(act_in) → act_blocks`;含 device ActSource 解码 + amax + 向偶数量化 + clamp;host 入口 `act_quant_run_host(...)` |
| `qmac.cuh` | dot 收敛为 WeightCodec::dot(保持 `RGD_HD`);ActReader = 现 `rgd_act_at`/`rgd_act_block` |
| `gemm.cuh/.cu` | 通用 GEMM kernel(读 act_blocks + weight_blocks),在片 OutConv 产 6 输出;`gemm_run_host` 签名加输出 buffer 组;dispatch runtime |
| `intgemm.cu` | 输出转换改用 `convert.cuh` 的 OutConv(int32 路径) |
| `reference.cpp/.h` | golden 读 act_blocks(double)验正确性(已基本如此;确认签名) |
| `dumper.cpp/.h` | 输出转换从 dumper **移除**(改由 kernel 产出后直接落盘);act_src 落盘保留;meta 更新 |
| `datagen.cpp/.h` | `datagen_fill` 仍产 fp 源(A 按 act_in round);`quantize_act` 可保留为 host 备用/校验,或废弃(act_blocks 改由 device 产) |
| `main.cpp` | 串联:datagen → 权重离线量化/reorder → **stage-1 device 量化** → stage-2 GEMM(产多输出)→ golden → dump |
| `docs/DESIGN.md` | 同步三级流水、黑盒接口、产物契约 |

## 5. 执行顺序(保持每步可编译)

1. `fp8.h`:decode 加 `RGD_HD`(host 仍可用)。**[加性,安全]**
2. `convert.cuh`:OutConv(`RGD_HD`),`rne_lround` 从 dumper 迁入;dumper 暂时仍调用它(行为不变)。**[重构,行为不变]**
3. `intgemm.cu`:输出转换改用 `convert.cuh`(行为不变)。验证 IntBlock 跑通。
4. `actquant.cuh/.cu`:stage-1 device 量化 kernel + `act_quant_run_host`;main 改为调用它产 act_blocks(替换 host `quantize_act`)。跑 legacy/k-quant 验正确性(对比旧 act_blocks 应一致/接近)。
5. `gemm.cu`:GEMM kernel 末尾加在片 OutConv,直接写 6 个 dtype 输出 buffer;`gemm_run_host` 增加输出 buffer 参数并拷回。
6. `dumper.cpp`:删除输出转换循环,改为直接落盘 kernel 产出的 6 buffer;meta 更新。
7. `main.cpp`:整合 stage-1/stage-2 新签名;case 名不变。
8. 全组合回归:A16/A8/A4 × W8(q8_1_64s/q8_0_64)/W4(q4_0_64) × act_in(F16/BF16/E4M3/E5M2/F32);K-quant(Q5_K_64S/Q6_K_64);IntBlock。校验 golden + check 脚本。
9. `DESIGN.md` 更新;全尺寸交付重跑。

## 6. 验证口径

- GPU 采数(fp32 acc)vs CPU golden(double):`max_abs ~1e-5`(可接受,以 GPU 为准)。
- dequant golden(reex `dequantize_row_*` + 浮点 matmul,psum=0):独立交叉校验若干行。
- `check_q5k64s_hw.py`:K-quant HW bit-pack 解码 + de-tile + 复算(PASS)。
- IntBlock:整数精确比较 `max_abs==0`。

## 7. 决策记录(用户已确认)

1. 激活在线量化吃 `act_src`(芯片内量化);权重离线量化。
2. 在线量化采用**独立量化 kernel(芯片第一级)→ 通用 GEMM kernel**(经 llama.cpp `from_float→wdata→vec_dot` 佐证)。
3. 激活 block **不带 sum**(全对称,点积用不上)。
4. 在线量化舍入 = **半数向偶数**(与输出统一)。
5. 输出 INT 饱和 = **两补码全范围** `[-(qmax+1),qmax]`。
6. IntBlock:**共享 OutConv、计算路径独立**(不强行并入一个大 kernel)。
7. **不要求**与 CPU 逐 bit(故不关 FMA;golden 仅验正确性)。
8. 实现策略:运行期 tag + kernel 内 switch(权重/A_bits/输出)。
