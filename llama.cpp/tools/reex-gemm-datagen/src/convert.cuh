// OutConv black box — fp32/int32 accumulator -> output dtype bytes.
// host+device (RGD_HD). Shared by the GEMM kernel (in-chip conversion), the
// IntBlock kernel and the dumper, so output-dtype logic lives in ONE place.
//   F16/BF16 : round-to-nearest-even cast (FP16 overflows to ±Inf).
//   E4M3     : 1-4-3 fp8, RNE + saturate to ±448 (no Inf encoding).
//   SInt (I16/I8/I6/I4): round-half-to-even (fp32 src) or exact (int32 src),
//              then saturate to two's-complement [-(qmax+1), qmax]. I6/I4 in int8.
//   UInt (U16/U8/U6/U4): same rounding then clamp to [0, qmax] (negatives -> 0).
//              U6/U4 in a uint8 container (zero-extended). All stores little-endian.
//   Output is pure saturation, NO scale.
#pragma once

#include "reex_layout.h"      // OutDType, OutSpec, OutKind, out_specs(), RGD_HD
#include "fp8.h"       // fp8_encode (E4M3), host+device
#include "ggml.h"      // host fp16/bf16 conversion

#if defined(__CUDACC__)
#include <cuda_fp16.h>
#endif

#include <cmath>
#include <cstdint>

namespace rgd {

RGD_HD inline uint16_t rgd_f32_to_f16_bits(float v) {
#if defined(__CUDA_ARCH__)
    return __half_as_ushort(__float2half_rn(v));
#else
    ggml_fp16_t h = ggml_fp32_to_fp16(v); return (uint16_t) h;
#endif
}

RGD_HD inline uint16_t rgd_f32_to_bf16_bits(float v) {
#if defined(__CUDA_ARCH__)
    uint32_t u = __float_as_uint(v);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x0040u); // NaN
    u += 0x7fffu + ((u >> 16) & 1u);                                              // RNE
    return (uint16_t) (u >> 16);
#else
    ggml_bf16_t h = ggml_fp32_to_bf16(v); uint16_t b; __builtin_memcpy(&b, &h, 2); return b;
#endif
}

// Round-half-to-even (banker's), independent of the runtime FP rounding mode.
RGD_HD inline long rgd_rne_lround(float x) {
    const double v  = (double) x;
    const double fl = floor(v);
    const double fr = v - fl;
    const long   lo = (long) fl;
    if (fr < 0.5) return lo;
    if (fr > 0.5) return lo + 1;
    return (lo & 1L) ? lo + 1 : lo;     // exact tie -> even
}

RGD_HD inline void rgd_store16(uint8_t * dst, uint16_t b) {
#if defined(__CUDA_ARCH__)
    // nvcc sm_120: byte stores from fp16/u16 can lower to st.global.u8,b16 and drop
    // the low byte (observed: 0x260C -> [0x00,0x26]). 16-bit store preserves LE.
    *reinterpret_cast<uint16_t *>(dst) = b;
#else
    dst[0] = (uint8_t) (b & 0xff);
    dst[1] = (uint8_t) (b >> 8);
#endif
}

// Store an already-rounded integer accumulator into a signed/unsigned int
// container of `bytes` with saturation. signed -> [-(qmax+1), qmax];
// unsigned -> [0, qmax]. 1-byte containers carry I6/I4 (sign-ext) / U6/U4.
RGD_HD inline void rgd_store_int(uint8_t * dst, OutKind kind, int bytes, int qmax, long r) {
    if (kind == OutKind::UInt) {
        if (r < 0)         r = 0;
        if (r > (long) qmax) r = qmax;
        if (bytes == 2) rgd_store16(dst, (uint16_t) r);
        else            dst[0] = (uint8_t) (unsigned) r;
    } else { // SInt
        const long qhi = qmax, qlo = -(long) (qmax + 1);
        if (r > qhi) r = qhi;
        if (r < qlo) r = qlo;
        if (bytes == 2) rgd_store16(dst, (uint16_t) (int16_t) r);
        else            dst[0] = (uint8_t) (int8_t) r;
    }
}

// fp32 accumulator -> one output dtype (quant / float-result path).
RGD_HD inline void rgd_out_write_f32(uint8_t * dst, OutKind kind,
                                     int bytes, int qmax, float v) {
    switch (kind) {
        case OutKind::F16:     rgd_store16(dst, rgd_f32_to_f16_bits(v));  break;
        case OutKind::BF16:    rgd_store16(dst, rgd_f32_to_bf16_bits(v)); break;
        case OutKind::FP8E4M3: dst[0] = fp8_encode(v, fp8_e4m3());        break;
        default:               rgd_store_int(dst, kind, bytes, qmax, rgd_rne_lround(v)); break;
    }
}

// int32 accumulator -> one output dtype (IntBlock pure-integer path: exact,
// no rounding; float = exact cast; int = clamp/saturate).
RGD_HD inline void rgd_out_write_i32(uint8_t * dst, OutKind kind,
                                     int bytes, int qmax, int32_t v) {
    switch (kind) {
        case OutKind::F16:     rgd_store16(dst, rgd_f32_to_f16_bits((float) v));  break;
        case OutKind::BF16:    rgd_store16(dst, rgd_f32_to_bf16_bits((float) v)); break;
        case OutKind::FP8E4M3: dst[0] = fp8_encode((float) v, fp8_e4m3());        break;
        default:               rgd_store_int(dst, kind, bytes, qmax, (long) v);   break;
    }
}

// Device-side fan-out: write all output dtypes for element `idx`. `bufs` holds
// the output base pointers in out_specs() order:
//   F16,BF16,E4M3, I16,I8,I6,I4, U16,U8,U6,U4
struct OutBufs { uint8_t * p[RGD_MAX_OUT]; };

RGD_HD inline void rgd_write_all_f32(const OutBufs & b, int64_t idx, float v) {
    rgd_out_write_f32(b.p[0]  + (size_t) idx * 2, OutKind::F16,     2,     0, v);
    rgd_out_write_f32(b.p[1]  + (size_t) idx * 2, OutKind::BF16,    2,     0, v);
    rgd_out_write_f32(b.p[2]  + (size_t) idx * 1, OutKind::FP8E4M3, 1,     0, v);
    rgd_out_write_f32(b.p[3]  + (size_t) idx * 2, OutKind::SInt,    2, 32767, v);
    rgd_out_write_f32(b.p[4]  + (size_t) idx * 1, OutKind::SInt,    1,   127, v);
    rgd_out_write_f32(b.p[5]  + (size_t) idx * 1, OutKind::SInt,    1,    31, v);
    rgd_out_write_f32(b.p[6]  + (size_t) idx * 1, OutKind::SInt,    1,     7, v);
    rgd_out_write_f32(b.p[7]  + (size_t) idx * 2, OutKind::UInt,    2, 65535, v);
    rgd_out_write_f32(b.p[8]  + (size_t) idx * 1, OutKind::UInt,    1,   255, v);
    rgd_out_write_f32(b.p[9]  + (size_t) idx * 1, OutKind::UInt,    1,    63, v);
    rgd_out_write_f32(b.p[10] + (size_t) idx * 1, OutKind::UInt,    1,    15, v);
}

RGD_HD inline void rgd_write_all_i32(const OutBufs & b, int64_t idx, int32_t v) {
    rgd_out_write_i32(b.p[0]  + (size_t) idx * 2, OutKind::F16,     2,     0, v);
    rgd_out_write_i32(b.p[1]  + (size_t) idx * 2, OutKind::BF16,    2,     0, v);
    rgd_out_write_i32(b.p[2]  + (size_t) idx * 1, OutKind::FP8E4M3, 1,     0, v);
    rgd_out_write_i32(b.p[3]  + (size_t) idx * 2, OutKind::SInt,    2, 32767, v);
    rgd_out_write_i32(b.p[4]  + (size_t) idx * 1, OutKind::SInt,    1,   127, v);
    rgd_out_write_i32(b.p[5]  + (size_t) idx * 1, OutKind::SInt,    1,    31, v);
    rgd_out_write_i32(b.p[6]  + (size_t) idx * 1, OutKind::SInt,    1,     7, v);
    rgd_out_write_i32(b.p[7]  + (size_t) idx * 2, OutKind::UInt,    2, 65535, v);
    rgd_out_write_i32(b.p[8]  + (size_t) idx * 1, OutKind::UInt,    1,   255, v);
    rgd_out_write_i32(b.p[9]  + (size_t) idx * 1, OutKind::UInt,    1,    63, v);
    rgd_out_write_i32(b.p[10] + (size_t) idx * 1, OutKind::UInt,    1,    15, v);
}

} // namespace rgd
