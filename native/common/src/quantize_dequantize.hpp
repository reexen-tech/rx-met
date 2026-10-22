// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cmath>

#ifdef __CUDACC__
#define RX_MET_QDQ_HOST_DEVICE __host__ __device__
#else
#define RX_MET_QDQ_HOST_DEVICE
#endif

namespace DlQuantization
{

RX_MET_QDQ_HOST_DEVICE inline float qdqReciprocal(float scale)
{
#ifdef __CUDA_ARCH__
    return __fdiv_rn(1.0f, scale);
#else
    return 1.0f / scale;
#endif
}

RX_MET_QDQ_HOST_DEVICE inline double qdqReciprocal(double scale)
{
#ifdef __CUDA_ARCH__
    return __ddiv_rn(1.0, scale);
#else
    return 1.0 / scale;
#endif
}

// Round in the centered grid before translating to TfEncoding's integer index
// grid. That translation cancels from QDQ, preserving low bits and tie parity.
template <typename T>
RX_MET_QDQ_HOST_DEVICE inline T stableQuantizeDequantize(T value, T minimum, T maximum, T scale,
                                                       bool stochastic, T noise)
{
    if (value != value)
        return value;
    const T clipped = fmax(fmin(value, maximum), minimum);
    // Match Torch's rounded reciprocal even when CUDA enables fast math.
    const T scaled = clipped * qdqReciprocal(scale);
    const T rounded = stochastic ? floor(scaled + noise) : nearbyint(scaled);
    return rounded * scale;
}

} // namespace DlQuantization

#undef RX_MET_QDQ_HOST_DEVICE
