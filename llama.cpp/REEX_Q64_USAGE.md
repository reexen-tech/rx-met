# REEX block-64 量化使用指南

极简版:如何调用不同量化类型 + 不同位宽截断精度。

## 1. 量化(选类型)

```bash
./build_cuda/bin/llama-quantize 输入-f16.gguf 输出.gguf <类型>
```

可选 `<类型>`(全部 block=64):

所有类型 block=64;`d`/`m`/`dmin` 均为 fp16。Legacy 每 64 元素一个 scale(无 sub-scale);
K-quant 超块=256、子块=64,全局 scale 量化子块 scale(scale 的 scale)。

| 类型 | 位宽 | 对/非对称 | 全局 scale | sub-scale(每 64) | min / 零点 |
|---|---|---|---|---|---|
| `Q4_0_64` | 4 | 对称 | fp16 | —(每 64 一个 scale) | 固定偏移 −8 |
| `Q5_0_64` | 5 | 对称 | fp16 | — | 固定偏移 −16 |
| `Q8_0_64` | 8 | 对称 | fp16 | — | 无 |
| `Q8_1_64` | 8 | 对称 | fp16 | — | 无(另存 fp16 和 `s`) |
| `Q4_1_64` | 4 | 非对称 | fp16 | — | fp16 `m` |
| `Q5_1_64` | 5 | 非对称 | fp16 | — | fp16 `m` |
| `Q2_K_64` | 2 | 非对称 | fp16 `d` | uint4 | fp16 `dmin` + uint4 |
| `Q4_K_64` | 4 | 非对称 | fp16 `d` | uint6 | fp16 `dmin` + uint6 |
| `Q5_K_64` | 5 | 非对称 | fp16 `d` | uint6 | fp16 `dmin` + uint6 |
| `Q3_K_64` | 3 | 对称 | fp16 `d` | int6 有符号 | 无(中点偏移) |
| `Q6_K_64` | 6 | 对称 | fp16 `d` | int8 有符号 | 无(中点偏移) |
| `Q2_K_64S` | 2 | 对称 | fp16 `d` | int4 有符号 | 无(中点偏移) |
| `Q4_K_64S` | 4 | 对称 | fp16 `d` | int6 有符号 | 无(中点偏移) |
| `Q5_K_64S` | 5 | 对称 | fp16 `d` | int6 有符号 | 无(中点偏移) |

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
