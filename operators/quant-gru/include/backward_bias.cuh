#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace kernel {

// 每个偏置只由一个线程块写回。固定分片和树形归约顺序，避免 atomicAdd 的
// 调度顺序改变浮点舍入结果，使固定输入的 QAT 梯度能够逐位复现。
template <typename T>
__global__ void ReduceBiasGradients(const int rows, const int features,
                                    const T *dp, const T *dq, T *dbw, T *dbr,
                                    const uint8_t *bw_mask, const uint8_t *br_mask) {
    __shared__ T sum_p[8][32];
    __shared__ T sum_q[8][32];
    const int feature = blockIdx.x * 32 + threadIdx.x;
    T p = static_cast<T>(0);
    T q = static_cast<T>(0);
    if (feature < features) {
        for (int row = threadIdx.y; row < rows; row += 8) {
            const size_t index = static_cast<size_t>(row) * features + feature;
            p += dp[index];
            q += dq[index];
        }
    }
    sum_p[threadIdx.y][threadIdx.x] = p;
    sum_q[threadIdx.y][threadIdx.x] = q;
    __syncthreads();
    for (int stride = 4; stride > 0; stride /= 2) {
        if (threadIdx.y < stride) {
            sum_p[threadIdx.y][threadIdx.x] += sum_p[threadIdx.y + stride][threadIdx.x];
            sum_q[threadIdx.y][threadIdx.x] += sum_q[threadIdx.y + stride][threadIdx.x];
        }
        __syncthreads();
    }
    if (threadIdx.y == 0 && feature < features) {
        // dp/dq 已包含线性输出和门的 mask；这里只应用偏置自身的量化 mask。
        dbw[feature] += sum_p[0][threadIdx.x] * static_cast<T>(bw_mask ? bw_mask[feature] : 1);
        dbr[feature] += sum_q[0][threadIdx.x] * static_cast<T>(br_mask ? br_mask[feature] : 1);
    }
}

}  // namespace kernel
