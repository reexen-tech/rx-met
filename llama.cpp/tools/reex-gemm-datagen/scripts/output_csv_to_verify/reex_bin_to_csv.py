#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""reex-gemm-datagen 的 .bin 采数结果 -> 十六进制 CSV 转换器（供硬件 model 复核）。

设计要点（详见 scripts/output_csv_to_verify/README.md）：
  * 读取一个 case 目录 + meta.json，自动识别 family(Legacy/Kquant/IntBlock)。
  * 全部数据 **de-tile** 回原始非 tiling 的二维矩阵后再写 CSV。
  * 每个元素写成 **原始 bit pattern 的十六进制**，按存储容器字节补零、小写、带 `0x` 前缀、
    负数按补码：int8/q4/I6/I4/U6/U4 -> 2 位；fp16/bf16/I16/U16 -> 4 位；e4m3 -> 2 位；
    f32/i32 -> 8 位。
  * 朝向与 zhuanhuan-yanzheng 对齐：weight_int=[K,N]、weight_scale=[K/64,N]、
    input_fp(=激活源)=[M,K]、act_int=[M,K]、act_scale=[M,K/agroup]、output=[M,N]。
  * Legacy 从 act_blocks.bin 导出片上量化结果 act_int + per-64 fp16 act_scale；
    Kquant 同样导出 act_int + BIN 原生的 per-256 fp16 act_scale。
    IntBlock 是纯整数路径，激活即 act_int（无 src、无 scale）。
  * weight scale 对应关系：weight 用 k//64 索引。
  * K-quant 的 scale 采用 weight_q6_k_scales.csv 的交错格式（每 super-block:
    1 行 fp super_scale + n_sub 行 int sub_scale），浮点同样写 hex。
  * 自校验：解码后取前若干行做 matmul 与 golden_f32 对比，打印 PASS/FAIL。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# hex lookup tables  (小写 + 0x 前缀)
# ---------------------------------------------------------------------------
_LUT8 = np.array([f"0x{i:02x}" for i in range(256)], dtype="<U4")
_LUT16 = np.array([f"0x{i:04x}" for i in range(1 << 16)], dtype="<U6")
# 无前缀数字表，供 u32 拼接后统一加 0x
_DIG16 = np.array([f"{i:04x}" for i in range(1 << 16)], dtype="<U4")


def hex_u8(a: np.ndarray) -> np.ndarray:
    return _LUT8[np.asarray(a, dtype=np.uint8)]


def hex_u16(a: np.ndarray) -> np.ndarray:
    return _LUT16[np.asarray(a, dtype=np.uint16)]


def hex_u32(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.uint32)
    hi = _DIG16[(a >> 16).astype(np.uint16)]
    lo = _DIG16[(a & 0xFFFF).astype(np.uint16)]
    return np.char.add("0x", np.char.add(hi, lo))


# native-dtype token -> (itemsize bytes, hex function on raw uint container)
def hex_by_dtype(raw_bytes_last: np.ndarray, token: str) -> np.ndarray:
    """raw_bytes_last: array whose last axis is the element's raw bytes (uint8, LE)."""
    t = token.upper()
    if t in ("E4M3", "E5M2", "I8", "U8", "I6", "U6", "I4", "U4"):
        return hex_u8(raw_bytes_last[..., 0])
    if t in ("F16", "BF16", "I16", "U16"):
        u = raw_bytes_last[..., 0].astype(np.uint16) | (raw_bytes_last[..., 1].astype(np.uint16) << 8)
        return hex_u16(u)
    if t in ("F32", "I32", "U32"):
        u = (
            raw_bytes_last[..., 0].astype(np.uint32)
            | (raw_bytes_last[..., 1].astype(np.uint32) << 8)
            | (raw_bytes_last[..., 2].astype(np.uint32) << 16)
            | (raw_bytes_last[..., 3].astype(np.uint32) << 24)
        )
        return hex_u32(u)
    raise ValueError(f"unknown dtype token {token}")


_DTYPE_BYTES = {
    "E4M3": 1, "E5M2": 1, "I8": 1, "U8": 1, "I6": 1, "U6": 1, "I4": 1, "U4": 1,
    "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
    "F32": 4, "I32": 4, "U32": 4,
}


# ---------------------------------------------------------------------------
# float decoders (only needed for self-check: weight/act from blocks + golden)
# ---------------------------------------------------------------------------
def f16_to_f32(u16: np.ndarray) -> np.ndarray:
    return u16.astype(np.uint16).view(np.float16).astype(np.float32)


# ---------------------------------------------------------------------------
# CSV writer (row-by-row to keep memory bounded)
# ---------------------------------------------------------------------------
def write_hex_csv(path: Path, str2d: np.ndarray) -> None:
    with open(path, "w", newline="") as f:
        for row in str2d.tolist():
            f.write(",".join(row))
            f.write("\n")


def write_hex_csv_rows(path: Path, rows: list) -> None:
    """rows: list of 1D str arrays (可含不同来源)，逐行写。"""
    with open(path, "w", newline="") as f:
        for r in rows:
            f.write(",".join(r.tolist()))
            f.write("\n")


# ---------------------------------------------------------------------------
# de-tile index builders
# ---------------------------------------------------------------------------
def idx_weight_src(N, K, Nt, Kt, Ntiles):
    n = np.arange(N)[:, None]
    k = np.arange(K)[None, :]
    return (((k // Kt) * Ntiles + n // Nt) * Nt + n % Nt) * Kt + k % Kt  # [N,K]


def idx_act_src(M, K, Mt, Kt, Ktiles):
    m = np.arange(M)[:, None]
    k = np.arange(K)[None, :]
    return (((m // Mt) * Ktiles + k // Kt) * Mt + m % Mt) * Kt + k % Kt  # [M,K]


def idx_result(M, N, Mt, Nt, Mtiles):
    m = np.arange(M)[:, None]
    n = np.arange(N)[None, :]
    return (n // Nt * Mtiles + m // Mt) * (Mt * Nt) + (m % Mt) * Nt + n % Nt  # [M,N]


def slot_weight_block(N, sb_count, Nt, Ntiles):
    n = np.arange(N)[:, None]
    sb = np.arange(sb_count)[None, :]
    return (sb * Ntiles + n // Nt) * Nt + n % Nt  # [N, sb_count]


def slot_act_block(M, kg_count, Mt, Ktiles):
    m = np.arange(M)[:, None]
    kg = np.arange(kg_count)[None, :]
    return (m // Mt * Ktiles + kg) * Mt + m % Mt  # [M, kg_count]


def decode_act_blocks(case, meta, M, K, Mt, Ktiles, agroup):
    """容器感知的 act 解码。返回:
       aq_MK  [M,K] int32 值
       aq_u   [M,K] uint(容器) 用于 hex
       ad_u16 [M,kg] fp16 bits
       ad_f   [M,kg] fp32 scale
       cont   容器字节数 (1 或 2)
    """
    abb = meta["files"]["act_blocks.bin"]["block_bytes"]
    act_bits = meta["quant"]["act_compute_bits"]
    cont = 2 if act_bits == 16 else 1
    kg_count = K // agroup
    ab = np.fromfile(case / "act_blocks.bin", dtype=np.uint8).reshape(-1, abb)
    aslot = slot_act_block(M, kg_count, Mt, Ktiles)
    ag = ab[aslot]                                            # [M,kg,abb]
    ad_u16 = ag[..., 0].astype(np.uint16) | (ag[..., 1].astype(np.uint16) << 8)
    ad_f = f16_to_f32(ad_u16)
    if cont == 1:
        aq_u = ag[..., 2:2 + agroup].astype(np.uint8)                 # [M,kg,agroup]
        aq = aq_u.view(np.int8).astype(np.int32)
        aq_u_full = aq_u.reshape(M, K)
    else:
        qb = ag[..., 2:2 + agroup * 2].reshape(ag.shape[0], ag.shape[1], agroup, 2)
        u = qb[..., 0].astype(np.uint16) | (qb[..., 1].astype(np.uint16) << 8)  # [M,kg,agroup]
        aq = u.astype(np.int16).astype(np.int32)
        aq_u_full = u.reshape(M, K)
    aq_MK = aq.reshape(M, K)
    return aq_MK, aq_u_full, ad_u16, ad_f, cont


# ---------------------------------------------------------------------------
# generic raw-byte de-tile -> hex string matrix + optional float matrix
# ---------------------------------------------------------------------------
def detile_elem_bytes(flat_bytes: np.ndarray, idx: np.ndarray, esize: int) -> np.ndarray:
    """flat_bytes: [total*esize] uint8;  idx: [R,C] element index.
    returns [R,C,esize] uint8 raw bytes per element."""
    buf = flat_bytes.reshape(-1, esize)      # [total, esize]
    return buf[idx]                           # [R,C,esize]


# ---------------------------------------------------------------------------
# LEGACY
# ---------------------------------------------------------------------------
def decode_legacy(case: Path, meta: dict, out: Path, outputs: list, check_rows: int):
    g, tl, q = meta["gemm"], meta["tiling"], meta["quant"]
    M, N, K = g["M"], g["N"], g["K"]
    Mt, Nt, Kt = tl["Mt"], tl["Nt"], tl["Kt"]
    Ntiles, Ktiles = N // Nt, K // Kt
    Mtiles = M // Mt
    agroup = tl["act_group_elems"]
    wtype = q["weight_type"]
    wbb = meta["files"]["weight_blocks.bin"]["block_bytes"]
    abb = meta["files"]["act_blocks.bin"]["block_bytes"]

    print(f"  family=Legacy wtype={wtype}  M,N,K={M},{N},{K}  Kt={Kt} agroup={agroup}")

    # ---- weight blocks ----
    wb = np.fromfile(case / "weight_blocks.bin", dtype=np.uint8).reshape(-1, wbb)
    wslot = slot_weight_block(N, Ktiles, Nt, Ntiles)          # [N,Ktiles]
    wg = wb[wslot]                                            # [N,Ktiles,wbb]
    d_u16 = wg[..., 0].astype(np.uint16) | (wg[..., 1].astype(np.uint16) << 8)  # [N,Ktiles]
    d_f = f16_to_f32(d_u16)

    if wtype == "q4_0_64":
        qs = wg[..., 2:2 + Kt // 2]                           # [N,Ktiles,32] uint8
        low = (qs & 0x0F).astype(np.int16)
        high = (qs >> 4).astype(np.int16)
        low = ((low ^ 0x08) - 0x08)
        high = ((high ^ 0x08) - 0x08)
        codes = np.concatenate([low, high], axis=-1)          # [N,Ktiles,64] in [-8,7]
    elif wtype == "q8_0_64":
        codes = wg[..., 2:2 + Kt].view(np.int8).astype(np.int16)
    elif wtype == "q8_1_64s":
        codes = wg[..., 4:4 + Kt].view(np.int8).astype(np.int16)
    else:
        raise ValueError(f"unsupported legacy wtype {wtype}")

    codes_NK = codes.reshape(N, K)                            # [N,K] value
    w_dq_NK = codes_NK.astype(np.float32) * np.repeat(d_f, Kt, axis=1)  # [N,K]

    # CSV: weight_int [K,N] hex(int8 container), weight_scale [Ktiles,N] hex(fp16)
    wint_hex_KN = hex_u8((codes_NK & 0xFF).astype(np.uint8)).T           # [K,N]
    wscale_hex = hex_u16(d_u16.T)                                        # [Ktiles,N]
    write_hex_csv(out / "weight_int.csv", wint_hex_KN)
    write_hex_csv(out / "weight_scale.csv", wscale_hex)
    print(f"  [w] weight_int.csv [{K},{N}]  weight_scale.csv [{Ktiles},{N}]")

    # ---- act blocks: 导出片上量化整数和原生 per-64 fp16 scale ----
    aq_MK, aq_u, ad_u16, ad_f, cont = decode_act_blocks(case, meta, M, K, Mt, Ktiles, agroup)
    if agroup != 64:
        raise ValueError(f"Legacy act_group_elems 应为 64，实际为 {agroup}")
    if cont == 1:
        write_hex_csv(out / "act_int.csv", hex_u8(aq_u))
    else:
        write_hex_csv(out / "act_int.csv", hex_u16(aq_u))
    write_hex_csv(out / "act_scale.csv", hex_u16(ad_u16))
    print(f"  [a] act_int.csv [{M},{K}]  act_scale.csv [{M},{K // agroup}] (per-{agroup}, fp16)")
    a_dq_MK = aq_MK.astype(np.float32) * np.repeat(ad_f, agroup, axis=1)

    # ---- source fp (激活即源精度 input_fp.csv, 无 scale) ----
    _dump_source_fp(case, meta, out, M, N, K, Mt, Nt, Kt, Ntiles, Ktiles)

    # ---- outputs ----
    _dump_outputs(case, meta, out, outputs, M, N, Mt, Nt, Mtiles)

    # ---- self-check ----
    _self_check(case, meta, a_dq_MK, w_dq_NK, M, N, Mt, Nt, Mtiles, check_rows)


# ---------------------------------------------------------------------------
# K-QUANT
# ---------------------------------------------------------------------------
_KQ_SCALE_BITS = {"Q6_K_64": 8, "Q5_K_64S": 6, "Q4_K_64S": 6, "Q3_K_64": 6, "Q2_K_64S": 4}


def _unpack_field(bits: np.ndarray, pos: int, width: int, signed: bool):
    """bits: [nb, total_bits] uint8 (LSB-first). 抽 [pos:pos+width] 每块一个整数。"""
    w = bits[:, pos:pos + width].astype(np.int64)
    val = w.dot(1 << np.arange(width, dtype=np.int64))
    if signed:
        val = np.where(val >= (1 << (width - 1)), val - (1 << width), val)
    return val


def decode_kquant(case: Path, meta: dict, out: Path, outputs: list, check_rows: int):
    g, tl, q = meta["gemm"], meta["tiling"], meta["quant"]
    M, N, K = g["M"], g["N"], g["K"]
    Mt, Nt, Kt = tl["Mt"], tl["Nt"], tl["Kt"]
    Ntiles, Ktiles = N // Nt, K // Kt
    Mtiles = M // Mt
    agroup = tl["act_group_elems"]
    wtype = q["weight_type"]
    wbits = q["weight_bits"]
    sbits = _KQ_SCALE_BITS[wtype]
    sub = 64
    n_sub = Kt // sub                                          # 4
    wbb = meta["files"]["weight_blocks.bin"]["block_bytes"]
    abb = meta["files"]["act_blocks.bin"]["block_bytes"]
    print(f"  family=Kquant wtype={wtype} wbits={wbits} sbits={sbits}  Kt={Kt} n_sub={n_sub}")

    wb = np.fromfile(case / "weight_blocks.bin", dtype=np.uint8).reshape(-1, wbb)
    wslot = slot_weight_block(N, Ktiles, Nt, Ntiles)          # [N,Ktiles]
    wg = wb[wslot].reshape(-1, wbb)                            # [N*Ktiles, wbb]
    nb = wg.shape[0]
    bits = np.unpackbits(wg, axis=1, bitorder="little")       # [nb, wbb*8]

    glb_u16 = _unpack_field(bits, 0, 16, signed=False).astype(np.uint16)   # [nb]
    glb_f = f16_to_f32(glb_u16)
    pos = 16
    sub_sc = np.zeros((nb, n_sub), dtype=np.int64)
    codes = np.zeros((nb, n_sub, sub), dtype=np.int64)
    for s in range(n_sub):
        sub_sc[:, s] = _unpack_field(bits, pos, sbits, signed=True)
        pos += sbits
        seg = bits[:, pos:pos + sub * wbits].reshape(nb, sub, wbits).astype(np.int64)
        c = seg.dot(1 << np.arange(wbits, dtype=np.int64))
        c = np.where(c >= (1 << (wbits - 1)), c - (1 << wbits), c)
        codes[:, s, :] = c
        pos += sub * wbits

    # reshape to [N, Ktiles, n_sub, sub]
    glb = glb_u16.reshape(N, Ktiles)                           # per super-block
    glb_fN = glb_f.reshape(N, Ktiles)
    sub_scN = sub_sc.reshape(N, Ktiles, n_sub)
    codesN = codes.reshape(N, Ktiles, n_sub, sub)
    codes_NK = codesN.reshape(N, K)

    # dequant W[n,k] = glb * sub_scale * code
    a_eff = (glb_fN[:, :, None] * sub_scN.astype(np.float32)).reshape(N, Ktiles * n_sub)  # [N, K/64]
    w_dq_NK = codes_NK.astype(np.float32) * np.repeat(a_eff, sub, axis=1)

    # CSV: weight_int [K,N]
    write_hex_csv(out / "weight_int.csv", hex_u8((codes_NK & 0xFF).astype(np.uint8)).T)

    # weight scale: interleaved (weight_q6_k_scales.csv 格式)
    #   每 super-block: 1 行 super_scale(fp16 hex) + n_sub 行 sub_scale(int hex), 列=N
    n_super = Ktiles
    rows = []
    for sup in range(n_super):
        rows.append(hex_u16(glb[:, sup]))                                  # [N] fp16
        for s in range(n_sub):
            rows.append(hex_u8((sub_scN[:, sup, s] & 0xFF).astype(np.uint8)))  # [N] int
    write_hex_csv_rows(out / "weight_scale.csv", rows)
    print(f"  [w] weight_int.csv [{K},{N}]  weight_scale.csv [{n_super*(1+n_sub)},{N}] (interleaved)")

    # ---- act blocks: 导出片上量化整数和 BIN 原生的 per-256 fp16 scale ----
    aq_MK, aq_u, ad_u16, ad_f, cont = decode_act_blocks(case, meta, M, K, Mt, Ktiles, agroup)
    if agroup != 256:
        raise ValueError(f"Kquant act_group_elems 应为 256，实际为 {agroup}")
    if cont == 1:
        write_hex_csv(out / "act_int.csv", hex_u8(aq_u))
    else:
        write_hex_csv(out / "act_int.csv", hex_u16(aq_u))
    write_hex_csv(out / "act_scale.csv", hex_u16(ad_u16))
    print(f"  [a] act_int.csv [{M},{K}]  act_scale.csv [{M},{K // agroup}] (per-{agroup}, fp16)")
    a_dq_MK = aq_MK.astype(np.float32) * np.repeat(ad_f, agroup, axis=1)

    _dump_source_fp(case, meta, out, M, N, K, Mt, Nt, Kt, Ntiles, Ktiles)
    _dump_outputs(case, meta, out, outputs, M, N, Mt, Nt, Mtiles)
    _self_check(case, meta, a_dq_MK, w_dq_NK, M, N, Mt, Nt, Mtiles, check_rows)


# ---------------------------------------------------------------------------
# INTBLOCK
# ---------------------------------------------------------------------------
def decode_intblock(case: Path, meta: dict, out: Path, outputs: list, check_rows: int):
    g, tl, q = meta["gemm"], meta["tiling"], meta["quant"]
    M, N, K = g["M"], g["N"], g["K"]
    Mt, Nt, Kt = tl["Mt"], tl["Nt"], tl["Kt"]
    Ntiles, Ktiles = N // Nt, K // Kt
    Mtiles = M // Mt
    wbits = int(q["weight_dtype"].split("-bit")[0].split()[-1])
    abits = int(q["act_dtype"].split("-bit")[0].split()[-1])
    print(f"  family=IntBlock wbits={wbits} abits={abits}  tile={Mt}x{Nt}x{Kt}")

    # weight: intra-block COLUMN(N)-major.  slot(n,k)=((k//16*Ntiles+n//16)*16+k%16)*16+n%16
    wbytes = np.fromfile(case / "weight_blocks.bin", dtype=np.uint8)
    wbits_stream = np.unpackbits(wbytes, bitorder="little")
    n = np.arange(N)[:, None]
    k = np.arange(K)[None, :]
    wslot = ((k // Kt * Ntiles + n // Nt) * Kt + k % Kt) * Nt + n % Nt   # [N,K]
    codes_NK = _gather_bits(wbits_stream, wslot, wbits, signed=True)

    # act: intra-block ROW(M)-major.  slot(m,k)=((m//16*Ktiles+k//16)*16+m%16)*16+k%16
    abytes = np.fromfile(case / "act_blocks.bin", dtype=np.uint8)
    abits_stream = np.unpackbits(abytes, bitorder="little")
    m = np.arange(M)[:, None]
    k2 = np.arange(K)[None, :]
    aslot = ((m // Mt * Ktiles + k2 // Kt) * Mt + m % Mt) * Kt + k2 % Kt  # [M,K]
    codes_MK = _gather_bits(abits_stream, aslot, abits, signed=True)

    # container hex: value clamped into signed byte/16b container
    if wbits <= 8:
        write_hex_csv(out / "weight_int.csv", hex_u8((codes_NK & 0xFF).astype(np.uint8)).T)
    else:
        write_hex_csv(out / "weight_int.csv", hex_u16((codes_NK & 0xFFFF).astype(np.uint16)).T)
    if abits <= 8:
        write_hex_csv(out / "act_int.csv", hex_u8((codes_MK & 0xFF).astype(np.uint8)))
    else:
        write_hex_csv(out / "act_int.csv", hex_u16((codes_MK & 0xFFFF).astype(np.uint16)))
    print(f"  [w] weight_int.csv [{K},{N}]   [a] act_int.csv [{M},{K}]  (IntBlock 无 scale)")

    _dump_outputs(case, meta, out, outputs, M, N, Mt, Nt, Mtiles)

    # self-check: pure int matmul vs golden (i32/f32)
    rows = min(check_rows, M)
    C = codes_MK[:rows].astype(np.float64) @ codes_NK.astype(np.float64).T
    _compare_golden(case, meta, C, rows, N, Mt, Nt, Mtiles)


def _gather_bits(stream: np.ndarray, slot: np.ndarray, width: int, signed: bool) -> np.ndarray:
    """stream: 1D unpackbits(LSB-first). slot[R,C]=element index. 抽 width bit 组成整数。"""
    base = (slot.astype(np.int64) * width)                    # [R,C]
    out = np.zeros(slot.shape, dtype=np.int64)
    for b in range(width):
        out += stream[base + b].astype(np.int64) << b
    if signed:
        out = np.where(out >= (1 << (width - 1)), out - (1 << width), out)
    return out


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _dump_source_fp(case, meta, out, M, N, K, Mt, Nt, Kt, Ntiles, Ktiles):
    files = meta["files"]
    # weight_src
    wsrc_name = "weight_src_f16.bin"
    wtok = files["weight_src_f16.bin"]["dtype"].upper()
    esz = _DTYPE_BYTES[wtok]
    wsrc = np.fromfile(case / wsrc_name, dtype=np.uint8)
    widx = idx_weight_src(N, K, Nt, Kt, Ntiles)               # [N,K]
    wb_bytes = detile_elem_bytes(wsrc, widx, esz)             # [N,K,esz]
    write_hex_csv(out / "weight_fp.csv", hex_by_dtype(wb_bytes, wtok).T)  # [K,N]

    # act_src (dtype varies)
    akey = [k for k in files if k.startswith("act_src_")][0]
    atok = files[akey]["dtype"].upper()
    esz_a = _DTYPE_BYTES[atok]
    asrc = np.fromfile(case / akey, dtype=np.uint8)
    aidx = idx_act_src(M, K, Mt, Kt, Ktiles)                  # [M,K]
    ab_bytes = detile_elem_bytes(asrc, aidx, esz_a)
    write_hex_csv(out / "input_fp.csv", hex_by_dtype(ab_bytes, atok))     # [M,K]
    print(f"  [src] weight_fp.csv [{K},{N}] ({wtok})  input_fp.csv [{M},{K}] ({atok})")


def _dump_outputs(case, meta, out, outputs, M, N, Mt, Nt, Mtiles):
    ridx = idx_result(M, N, Mt, Nt, Mtiles)                   # [M,N]
    avail = sorted(p.name for p in case.glob("output_*.bin"))
    for fn in avail:
        tok = fn[len("output_"):-len(".bin")].upper()
        if outputs != ["ALL"] and tok not in outputs:
            continue
        if tok not in _DTYPE_BYTES:
            continue
        esz = _DTYPE_BYTES[tok]
        raw = np.fromfile(case / fn, dtype=np.uint8)
        ob = detile_elem_bytes(raw, ridx, esz)               # [M,N,esz]
        write_hex_csv(out / f"output_{tok}.csv", hex_by_dtype(ob, tok))
    print(f"  [out] wrote {len([f for f in avail])} output_*.csv (filtered)")


def _self_check(case, meta, a_dq_MK, w_dq_NK, M, N, Mt, Nt, Mtiles, check_rows):
    rows = min(check_rows, M)
    C = a_dq_MK[:rows].astype(np.float64) @ w_dq_NK.astype(np.float64).T   # [rows,N]
    _compare_golden(case, meta, C, rows, N, Mt, Nt, Mtiles)


def _compare_golden(case, meta, C, rows, N, Mt, Nt, Mtiles):
    gpath = case / "golden_f32.bin"
    if not gpath.exists():
        print("  [check] golden_f32.bin 缺失, 跳过校验")
        return
    gold = np.fromfile(gpath, dtype=np.float32)
    ridx = idx_result(rows, N, Mt, Nt, Mtiles)
    G = gold[ridx].astype(np.float64)
    err = np.abs(C - G)
    den = np.maximum(np.abs(G), 1e-6)
    max_abs = float(err.max())
    max_rel = float((err / den).max())
    tol = 1e-2 * max(1.0, float(np.abs(G).max()))
    ok = max_abs < tol
    print(f"  [check] rows={rows} max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
          f"tol={tol:.3e} -> {'PASS' if ok else 'FAIL'}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def convert_case(case: Path, out_root: Path, outputs: list, check_rows: int):
    meta = json.loads((case / "meta.json").read_text())
    fam = meta["quant"].get("family") or meta["quant"].get("weight_family")
    out = out_root / case.name
    out.mkdir(parents=True, exist_ok=True)
    print(f"[case] {case.name}  family={fam}  -> {out}")
    t0 = time.time()
    if fam == "Legacy":
        decode_legacy(case, meta, out, outputs, check_rows)
    elif fam == "Kquant":
        decode_kquant(case, meta, out, outputs, check_rows)
    elif fam == "IntBlock":
        decode_intblock(case, meta, out, outputs, check_rows)
    else:
        raise ValueError(f"unknown family {fam}")
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[done] {case.name}  {time.time()-t0:.1f}s\n")


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="reex-gemm-datagen .bin -> hex CSV")
    ap.add_argument("case", nargs="+", help="case 目录 (含 meta.json)")
    ap.add_argument("--out", default=str(here / "output"), help="输出根目录")
    ap.add_argument("--outputs", default="ALL",
                    help="要导出的 output dtype, 逗号分隔 (默认 ALL; none=不导)")
    ap.add_argument("--check-rows", type=int, default=64, help="自校验行数")
    args = ap.parse_args()

    if args.outputs.upper() == "NONE":
        outputs = []
    elif args.outputs.upper() == "ALL":
        outputs = ["ALL"]
    else:
        outputs = [t.strip().upper() for t in args.outputs.split(",")]

    out_root = Path(args.out)
    for c in args.case:
        convert_case(Path(c), out_root, outputs, args.check_rows)


if __name__ == "__main__":
    main()
