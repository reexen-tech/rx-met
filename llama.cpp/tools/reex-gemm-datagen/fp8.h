// Minimal FP8 (E4M3 / E5M2) emulation for activation source dtypes.
//   - E4M3: 1-4-3, bias 7, max-normal 448, NO inf (S.1111.111 = NaN).
//   - E5M2: 1-5-2, bias 15, max-normal 57344, IEEE-like inf/NaN.
// Round-to-nearest-even via nearbyintf (default FP rounding mode). Host-only.
#pragma once

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
    #define RGD_FP8_HD __host__ __device__
#else
    #define RGD_FP8_HD
#endif

namespace rgd {

struct Fp8Fmt { int ebits, mbits, bias; float max_normal; bool has_inf; };

// host+device formats (constexpr so they can be used inside device code too).
RGD_FP8_HD inline Fp8Fmt fp8_e4m3() { return Fp8Fmt{4, 3,  7,   448.0f, false}; }
RGD_FP8_HD inline Fp8Fmt fp8_e5m2() { return Fp8Fmt{5, 2, 15, 57344.0f, true }; }

// Decode an 8-bit code back to fp32 (host+device; uses global ldexpf).
RGD_FP8_HD inline float fp8_decode(uint8_t code, const Fp8Fmt & f) {
    const int     maxexp = (1 << f.ebits) - 1;
    const uint8_t sign   = (code >> (f.ebits + f.mbits)) & 1u;
    const int     exp    = (code >> f.mbits) & ((1 << f.ebits) - 1);
    const int     man    = code & ((1 << f.mbits) - 1);
    float val;
    if (exp == maxexp && (f.has_inf || man == (1 << f.mbits) - 1)) {
        val = (f.has_inf && man == 0) ? INFINITY : NAN;               // inf/NaN
    } else if (exp == 0) {
        val = ldexpf((float) man, 1 - f.bias - f.mbits);              // subnormal
    } else {
        val = ldexpf((float) ((1 << f.mbits) | man), exp - f.bias - f.mbits);
    }
    return sign ? -val : val;
}

// Encode fp32 -> 8-bit code (RNE, saturating).
inline uint8_t fp8_encode(float x, const Fp8Fmt & f) {
    const int     maxexp = (1 << f.ebits) - 1;
    const uint8_t sign   = std::signbit(x) ? 1u : 0u;
    const uint8_t scode  = (uint8_t) (sign << (f.ebits + f.mbits));
    const auto    inf_code = [&]() { return (uint8_t) (scode | (((1 << f.ebits) - 1) << f.mbits)); };
    const auto    max_code = [&]() { return (uint8_t) (scode | (maxexp << f.mbits) | ((1 << f.mbits) - 2)); };

    if (std::isnan(x)) return (uint8_t) (scode | (maxexp << f.mbits) | ((1 << f.mbits) - 1));
    const float ax = std::fabs(x);
    if (ax == 0.0f) return scode;
    if (ax > f.max_normal) return f.has_inf ? inf_code() : max_code();

    int e2; std::frexp(ax, &e2);
    const int E    = e2 - 1;            // unbiased exponent of leading 1
    const int emin = 1 - f.bias;
    if (E < emin) {                      // subnormal target
        const float step = std::ldexp(1.0f, emin - f.mbits);
        int man = (int) std::nearbyint(ax / step);
        if (man >= (1 << f.mbits)) return (uint8_t) (scode | (1 << f.mbits)); // -> smallest normal
        return (uint8_t) (scode | man);
    }
    const float step = std::ldexp(1.0f, E - f.mbits);
    int q   = (int) std::nearbyint(ax / step);     // [2^m, 2^(m+1)]
    int man = q - (1 << f.mbits);
    int ef  = E + f.bias;
    if (man >= (1 << f.mbits)) { man = 0; ++ef; }   // mantissa carry
    if (f.has_inf) {
        if (ef >= maxexp) return inf_code();
    } else {
        if (ef > maxexp || (ef == maxexp && man > (1 << f.mbits) - 2)) return max_code();
    }
    return (uint8_t) (scode | (ef << f.mbits) | man);
}

inline float fp8_round(float x, const Fp8Fmt & f) { return fp8_decode(fp8_encode(x, f), f); }

} // namespace rgd
