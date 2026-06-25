// OutConv black box — fp32/int32 accumulator -> output dtype bytes.
// host+device (RGD_HD). Shared by the GEMM kernel (in-chip conversion), the
// IntBlock kernel and the dumper, so output-dtype logic lives in ONE place.
//   float (F16/BF16): round-to-nearest-even cast.
//   int   (I16/I8/I6/I4): round-half-to-even (fp32 src) or exact (int32 src),
//                         then saturate to two's-complement [-(qmax+1), qmax].
//   I6/I4 use an int8 container (sign-extended). All stores little-endian.
#pragma once

#include "case.h"      // OutDType, OutSpec, out_specs(), RGD_HD
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
    dst[0] = (uint8_t) (b & 0xff);
    dst[1] = (uint8_t) (b >> 8);
}

// fp32 accumulator -> one output dtype (quant / float-result path).
RGD_HD inline void rgd_out_write_f32(uint8_t * dst, OutDType dt, bool is_float,
                                     int bytes, int qmax, float v) {
    if (is_float) {
        const uint16_t b = (dt == OutDType::F16) ? rgd_f32_to_f16_bits(v)
                                                 : rgd_f32_to_bf16_bits(v);
        rgd_store16(dst, b);
    } else {
        long r = rgd_rne_lround(v);
        const long qhi = qmax, qlo = -(long) (qmax + 1);
        if (r > qhi) r = qhi;
        if (r < qlo) r = qlo;
        if (bytes == 2) rgd_store16(dst, (uint16_t) (int16_t) r);
        else            dst[0] = (uint8_t) (int8_t) r;
    }
}

// int32 accumulator -> one output dtype (IntBlock pure-integer path: exact,
// no rounding; float = exact cast; int = clamp/saturate).
RGD_HD inline void rgd_out_write_i32(uint8_t * dst, OutDType dt, bool is_float,
                                     int bytes, int qmax, int32_t v) {
    if (is_float) {
        const uint16_t b = (dt == OutDType::F16) ? rgd_f32_to_f16_bits((float) v)
                                                 : rgd_f32_to_bf16_bits((float) v);
        rgd_store16(dst, b);
    } else {
        long r = v;
        const long qhi = qmax, qlo = -(long) (qmax + 1);
        if (r > qhi) r = qhi;
        if (r < qlo) r = qlo;
        if (bytes == 2) rgd_store16(dst, (uint16_t) (int16_t) r);
        else            dst[0] = (uint8_t) (int8_t) r;
    }
}

// Device-side fan-out: write all 6 dtypes for element `idx`. `bufs` holds the 6
// output base pointers in out_specs() order: F16,BF16,I16,I8,I6,I4.
struct OutBufs { uint8_t * p[6]; };

RGD_HD inline void rgd_write_all_f32(const OutBufs & b, int64_t idx, float v) {
    rgd_out_write_f32(b.p[0] + (size_t) idx * 2, OutDType::F16,  true,  2,     0, v);
    rgd_out_write_f32(b.p[1] + (size_t) idx * 2, OutDType::BF16, true,  2,     0, v);
    rgd_out_write_f32(b.p[2] + (size_t) idx * 2, OutDType::I16,  false, 2, 32767, v);
    rgd_out_write_f32(b.p[3] + (size_t) idx * 1, OutDType::I8,   false, 1,   127, v);
    rgd_out_write_f32(b.p[4] + (size_t) idx * 1, OutDType::I6,   false, 1,    31, v);
    rgd_out_write_f32(b.p[5] + (size_t) idx * 1, OutDType::I4,   false, 1,     7, v);
}

RGD_HD inline void rgd_write_all_i32(const OutBufs & b, int64_t idx, int32_t v) {
    rgd_out_write_i32(b.p[0] + (size_t) idx * 2, OutDType::F16,  true,  2,     0, v);
    rgd_out_write_i32(b.p[1] + (size_t) idx * 2, OutDType::BF16, true,  2,     0, v);
    rgd_out_write_i32(b.p[2] + (size_t) idx * 2, OutDType::I16,  false, 2, 32767, v);
    rgd_out_write_i32(b.p[3] + (size_t) idx * 1, OutDType::I8,   false, 1,   127, v);
    rgd_out_write_i32(b.p[4] + (size_t) idx * 1, OutDType::I6,   false, 1,    31, v);
    rgd_out_write_i32(b.p[5] + (size_t) idx * 1, OutDType::I4,   false, 1,     7, v);
}

} // namespace rgd
