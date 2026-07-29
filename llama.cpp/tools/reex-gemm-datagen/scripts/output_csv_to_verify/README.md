# output_csv_to_verify — reex-gemm-datagen 采数结果转十六进制 CSV

把 `reex-gemm-datagen` 生成的二进制 case（`.bin` + `meta.json`）**de-tile 回原始非 tiling 布局**，
每个元素以**原始 bit pattern 的十六进制**写成 CSV，供硬件 model 做 bit 级复核。
scale 的对应方式对齐 `tools/zhuanhuan-yanzheng`。

## 用法

```bash
# 单个 case
python3 reex_bin_to_csv.py <case目录> [--out DIR] [--outputs ALL|none|F16,I8,...] [--check-rows R]

# 批量转 output/legacy 下全部 case（默认输出到本目录 output/）
bash run_legacy.sh
#   env 覆盖: SRC=<case根目录> OUT=<输出根> OUTPUTS=<ALL|none|...> CHECK_ROWS=64
```

产物默认写到 `scripts/output_csv_to_verify/output/<case名>/`，每个 case 一个子目录，
并附带原始 `meta.json`。

## 支持的 family

自动读 `meta.json` 识别，无需指定：

| family | 权重类型 | scale |
|---|---|---|
| Legacy | `q4_0_64` / `q8_0_64` / `q8_1_64s` | 单级 per-64 `d`(fp16) |
| Kquant | `Q2/Q3/Q4/Q5/Q6_K_64(S)` | super(fp16) + sub(int)，交错格式 |
| IntBlock | 纯整数 `INT` | 无 |

## 输出文件（对齐 zhuanhuan-yanzheng）

| 文件 | 形状 | 内容 |
|---|---|---|
| `weight_int.csv` | `[K, N]` | 量化权重码（hex，容器字节补码） |
| `weight_scale.csv` | Legacy `[K/64, N]`；K-quant 交错 `[(K/256)*(1+n_sub), N]`；INT 无 | scale（hex） |
| `weight_fp.csv` | `[K, N]` | 量化前源权重（fp16 hex） |
| `input_fp.csv` | `[M, K]` | **激活输入**（源精度 hex：F16/BF16/E4M3/E5M2/F32）。激活无 scale |
| `act_int.csv` | Legacy/K-quant/INT `[M, K]` | 激活量化整数码（hex，容器字节补码） |
| `act_scale.csv` | Legacy `[M, K/64]`；K-quant `[M, K/256]` | 每个激活量化组共用的 fp16 scale（原始 bit pattern hex） |
| `output_<DT>.csv` × 11 | `[M, N]` | 11 种 dtype 的 GPU 结果（hex） |

> Legacy 激活从 `act_blocks.bin` 解码，同时导出片上量化后的 `act_int.csv` 和 per-64 `act_scale.csv`。
> K-quant 同样从 `act_blocks.bin` 解码，按其原生布局导出 `act_int.csv` 和 per-256 `act_scale.csv`。
> IntBlock 是纯整数路径，激活即 `act_int.csv`（无 src、无 scale）。

## 十六进制约定

- **原始 bit pattern**，**小写、带 `0x` 前缀**。
- **K-quant 权重码 / sub_scale**、Legacy `q4_0_64` 码、IntBlock 窄整数：按 **实际量化位宽**
  写 hex（有符号补码原码，**不符号扩展到 8 bit**）。例如 Q2 sub_scale `-8` → `0x8`（非 `0xf8`）；
  Q3/Q4/Q5 sub_scale `-32` → `0x20`（非 `0xe0`）。
- 仍按**存储容器字节**写的：`act_int`(A8/A4/A16 容器)、Legacy q8、fp16/bf16/I16/U16(4 hex)、
  e4m3/e5m2/int8 容器(2 hex)、f32/i32(8 hex)。
- 示例：fp16 `1.0` → `0x3c00`；A8 `-8` → `0xf8`；Q2 sub_scale `-8` → `0x8`；Q4 sub_scale `-1` → `0x3f`。

## scale 对应关系

- 权重：`W_dq[k,n] = weight_int[k,n] * weight_scale[k//64, n]`（K-quant 再乘上层 super_scale）。
- Legacy 激活：`A_dq[m,k] = act_int[m,k] * fp16(act_scale[m,k//64])`。
- K-quant 激活：`A_dq[m,k] = act_int[m,k] * fp16(act_scale[m,k//256])`。
- `input_fp.csv` 仍保留量化前的源精度激活。
- K-quant `weight_scale.csv` 采用 `scripts/weight_q6_k_scales.csv` 的交错布局：
  每个 super-block 占 `1 + n_sub` 行 —— 第 1 行 fp16 super_scale，其后 `n_sub` 行 int sub_scale，列为 N。

## 自校验

解码后取前 `--check-rows` 行做 `A_dq @ W_dq^T`（IntBlock 为纯整数），与 `golden_f32.bin`（de-tile 后）
对比，打印 `max_abs / max_rel` 与 `PASS/FAIL`（阈值随 golden 量级自适应）。全部 11 个 legacy case 均 PASS，
误差仅量化/精度级（~1e-6）。

## de-tile 公式（来自各 case 的 meta.json）

- 权重源 `idx(n,k) = ((k/Kt*Ntiles + n/Nt)*Nt + n%Nt)*Kt + k%Kt`
- 激活源 `idx(m,k) = ((m/Mt*Ktiles + k/Kt)*Mt + m%Mt)*Kt + k%Kt`
- 结果   `idx(m,n) = (n/Nt*Mtiles + m/Mt)*(Mt*Nt) + (m%Mt)*Nt + n%Nt`
- 权重块 `slot(n,sb) = (sb*Ntiles + n/Nt)*Nt + n%Nt`
- 激活块 `slot(m,kg) = (m/Mt*Ktiles + kg)*Mt + m%Mt`
- IntBlock 位打包见 `reex_bin_to_csv.py::decode_intblock`。
