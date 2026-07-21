# REEX block-64 量化使用指南

极简版:如何调用不同量化类型 + 不同位宽截断精度。

## 0. 先编译 + 转 FP16 GGUF(必做)

```bash
# 1) 编译(先做)
# CPU
cd /path/to/llama.cpp
cmake -B build_cpu -DGGML_USE_REEX_Q64=ON
cmake --build build_cpu --config Release -j
# GPU(Q64 + LUT)
cmake -B build_cuda -DGGML_CUDA=ON \
  -DGGML_USE_REEX_Q64=ON \
  -DGGML_USE_REEX=ON
cmake --build build_cuda --config Release -j

# 2) 检查是否成功
./build_cuda/bin/llama-cli --help
./build_cuda/bin/llama-quantize --help

# 3) (可选) AWQ scale → FP16 HF；跳过则直接用原 HF
#     --w_bit 须与步骤 1 的量化位宽一致(例: w_bit=4 → Q4_*_64)
#     校准默认: pileval (mit-han-lab/pile-val-backup), n_samples=128, seqlen=512
python export_awq_hf.py \
  --model_path /path/to/model \
  --output_dir /path/to/out-awq-hf \
  --w_bit 4 --q_group_size 64 \
  --n_samples 128 --seqlen 512 --calib_data pileval # 使用pileval数据集进行校准， 校准样本128个， 序列长度512.

# 4) HF → FP16 GGUF
python convert_hf_to_gguf.py /path/to/hf_or_awq_hf \
  --outtype f16 --outfile /path/to/model-f16.gguf
```

流程: `HF (或 AWQ-HF) → f16.gguf → llama-quantize → Qx_*.gguf`。

## 1. 量化(选类型)

```bash
# 默认:主体为 <类型>,output.weight 常抬到 Q6_K_64
./build_cuda/bin/llama-quantize 输入-f16.gguf 输出.gguf <类型>

# 全张量同一类型(对照 / 严格 W4 时用)
./build_cuda/bin/llama-quantize --pure 输入-f16.gguf 输出.gguf <类型>
```

可选 `<类型>`(全部 block=64):

所有类型 block=64;`d`/`m`/`dmin` 均为 fp16。Legacy 每 64 元素一个 scale(无 sub-scale);
K-quant 超块=256、子块=64,全局 scale 量化子块 scale(scale 的 scale)。

> **存储语义(重要)**:除 Legacy 的非对称类型(`Q4_1_64`/`Q5_1_64`,带 fp16 `m`)
> 和 K-quant 的非对称类型(`Q2_K_64`/`Q4_K_64`/`Q5_K_64`,带 `dmin`+min)外,**所有对称量化的
> 权重(及对称 K-quant 的子块 scale)均以二进制补码(two's complement)有符号整数直接存储**,
> 反量化即 `x = d * q`(K-quant 为 `x = d * scale * q`),不再有任何 `q − mid` 零点偏移。
> 子字节(2/3/4/5/6 bit)按补码位段打包,读取时按位符号扩展 `(n^M)−M`(硬件免费),
> 不是算术减偏移。**此改动使旧 GGUF 数值不兼容,需重新量化。**

| 类型 | 位宽 | 对/非对称 | 全局 scale | sub-scale(每 64) | 权重存储 / min |
|---|---|---|---|---|---|
| `Q4_0_64` | 4 | 对称 | fp16 | —(每 64 一个 scale) | int4 补码 [−8,7],无零点 |
| `Q5_0_64` | 5 | 对称 | fp16 | — | int5 补码 [−16,15],无零点 |
| `Q8_0_64` | 8 | 对称 | fp16 | — | int8,无零点 |
| `Q8_1_64` | 8 | 对称 | fp16 | — | int8,无零点(另存 fp16 和 `s`) |
| `Q4_1_64` | 4 | 非对称 | fp16 | — | uint4 + fp16 `m` |
| `Q5_1_64` | 5 | 非对称 | fp16 | — | uint5 + fp16 `m` |
| `Q2_K_64` | 2 | 非对称 | fp16 `d` | uint4 | uint2 + fp16 `dmin` + uint4 min |
| `Q4_K_64` | 4 | 非对称 | fp16 `d` | uint6 | uint4 + fp16 `dmin` + uint6 min |
| `Q5_K_64` | 5 | 非对称 | fp16 `d` | uint6 | uint5 + fp16 `dmin` + uint6 min |
| `Q3_K_64` | 3 | 对称 | fp16 `d` | int6 补码 | int3 补码 [−4,3],无零点 |
| `Q6_K_64` | 6 | 对称 | fp16 `d` | int8 有符号 | int6 补码 [−32,31],无零点 |
| `Q2_K_64S` | 2 | 对称 | fp16 `d` | int4 补码 | int2 补码 [−2,1],无零点 |
| `Q4_K_64S` | 4 | 对称 | fp16 `d` | int6 补码 | int4 补码 [−8,7],无零点 |
| `Q5_K_64S` | 5 | 对称 | fp16 `d` | int6 补码 | int5 补码 [−16,15],无零点 |

## 2. 推理 / 跑分(选位宽截断)

用环境变量 `REEX_Q64_PSUM_BITS` 控制定点 Psum 截断到 B 位(不设或 `<=0` = 关闭):

```bash
# 不截断(默认)
./build_cuda/bin/llama-cli -m 模型.gguf -ngl 99 -p "hello" -st

# 截断到 8 位
REEX_Q64_PSUM_BITS=8 \
./build_cuda/bin/llama-perplexity -m 模型.gguf -ngl 99 \
    -f models/q64test/wikitext-2-raw/wiki.test.raw -c 512 --chunks 4
```

## 要点

- `-ngl 99` 走 GPU,`-ngl 0` 走 CPU,二者数值对齐(截断也对齐)。
- `B` 粒度统一为 **64 元素 Psum**;只截断整数 Psum,min 修正项不截断。
- 截断对所有类型、CPU 与 GPU(decode + prefill)同时生效;改 `B` 无需重新量化或重编译。



## 3. 实测(qwen2.5-0.5b)

```bash
M=models/q64test/qwen2.5-0.5b-Q4_K_64.gguf
F=models/q64test/wikitext-2-raw/wiki.test.raw

# 跑分(单个类型)
./build_cuda/bin/llama-perplexity -m $M -ngl 99 -f $F -c 512 --chunks 8

# 位宽截断(B=8)
REEX_Q64_PSUM_BITS=8 ./build_cuda/bin/llama-perplexity -m $M -ngl 99 -f $F -c 512 --chunks 8

# GPU≡CPU 对齐验证(同一 B 两边结果应完全一致)
REEX_Q64_PSUM_BITS=8 ./build_cuda/bin/llama-perplexity -m $M -ngl 99 -f $F -c 512 --chunks 8  # GPU
REEX_Q64_PSUM_BITS=8 ./build_cuda/bin/llama-perplexity -m $M -ngl 0  -f $F -c 512 --chunks 8  # CPU

# 批量对比所有类型
for t in Q8_0_64 Q8_1_64 Q4_0_64 Q5_0_64 Q4_1_64 Q5_1_64 \
         Q4_K_64 Q5_K_64 Q6_K_64 Q2_K_64 Q3_K_64 Q4_K_64S Q5_K_64S Q2_K_64S; do
  ppl=$(./build_cuda/bin/llama-perplexity -m models/q64test/qwen2.5-0.5b-$t.gguf -ngl 99 \
        -f $F -c 512 --chunks 8 2>&1 | grep -oE "PPL = [0-9.]+" | tail -1)
  printf "%-12s %s\n" "$t" "$ppl"
done
```

### 实测结果(wikitext-2,`-c 512 --chunks 8`,f16 基线 = 15.55)

| 类型 | PPL | 类型 | PPL |
|---|---|---|---|
| Q8_0_64 | 15.70 | Q4_K_64 | 18.29 |
| Q8_1_64 | 15.70 | Q5_K_64 | 16.19 |
| Q6_K_64 | 15.72 | Q4_K_64S | 18.16 |
| Q5_1_64 | 16.19 | Q5_K_64S | 16.83 |
| Q5_0_64 | 16.63 | Q3_K_64 | 18.53 |
| Q4_1_64 | 18.05 | Q2_K_64 | 25.24 |
| Q4_0_64 | 18.21 | Q2_K_64S | 33.68 |

Psum 截断(Q4_K_64):off=18.29 → B=12/10/8≈18.2~18.4 → B=6=19.92 → B=4 崩溃。
GPU(`-ngl 99`)与 CPU(`-ngl 0`)在 B=8 下均为 **18.3640**,完全对齐。

## 4. 实测(Qwen3.5-35B-A3B,4×4090 layer split)

```bash
F16=/path/Qwen3.5-35B-A3B-f16.gguf
F=models/q64test/wikitext-2-raw/wiki.test.raw

# 量化(MoE,每类型约 5~10 分钟)
./build_cuda/bin/llama-quantize $F16 models/q64test_35b/Qwen3.5-35B-A3B-Q4_K_64.gguf Q4_K_64

# 跑分(多卡)
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build_cuda/bin/llama-perplexity \
    -m models/q64test_35b/Qwen3.5-35B-A3B-Q4_K_64.gguf -ngl 99 -f $F -c 512 --chunks 8
```

### 结果(wikitext-2,`-c 512 --chunks 8`,f16 基线 = 6.1040,65G)

| 类型 | 大小 | PPL | 类型 | 大小 | PPL |
|---|---|---|---|---|---|
| Q8_0_64 | 34G | 6.1100 | Q6_K_64 | 26G | 6.1305 |
| Q8_1_64 | 35G | 6.1100 | Q5_K_64 | 22G | 6.1530 |
| Q5_0_64 | 22G | 6.1153 | Q4_K_64 | 18G | 6.4427 |
| Q5_1_64 | 23G | 6.2120 | Q4_K_64S | 18G | 6.4983 |
| Q4_0_64 | 18G | 6.4812 | Q3_K_64 | 6.1G | 8.0138 |
| Q4_1_64 | 19G | 6.6158 | Q2_K_64 | 9.5G | 566.38(崩溃) |

- 8/6/5-bit 近无损;4-bit 退化 ~6%;3-bit 勉强可用;2-bit 崩溃(MoE 纯 2-bit 需 imatrix)。
- 单卡 ≡ 多卡(layer split)结果一致;`--split-mode row` 会破坏 64 元素 Psum 对齐,勿用。

### 位宽截断(Q4_K_64)

```bash
M=models/q64test_35b/Qwen3.5-35B-A3B-Q4_K_64.gguf
REEX_Q64_PSUM_BITS=8 CUDA_VISIBLE_DEVICES=0,1,2,3 ./build_cuda/bin/llama-perplexity -m $M -ngl 99 -f $F -c 512 --chunks 8  # GPU
REEX_Q64_PSUM_BITS=8 CUDA_VISIBLE_DEVICES=""        ./build_cuda/bin/llama-perplexity -m $M -ngl 0  -f $F -c 512 --chunks 8  # CPU
```

| 配置 | PPL |
|---|---|
| 不截断 | 6.4427 |
| B=8 GPU | 6.4540 |
| B=8 CPU | 6.4768 |