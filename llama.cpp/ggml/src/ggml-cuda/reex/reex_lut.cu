#ifdef GGML_USE_REEX

#include "reex_lut.cuh"
#include "reex_lut_test.h"

#include <cuda_runtime.h>

static __global__ void ggml_cuda_reex_lut_test_kernel(
        const float * input, float * output, size_t count, int op) {
    const size_t index = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    if (op == GGML_CUDA_REEX_LUT_TEST_RECIPROCAL) {
        output[index] = ggml_cuda_reciprocal_lut_mixed_fp16_reex(input[index]);
    } else {
        output[index] = ggml_cuda_rsqrt_lut_mixed_fp16_reex(input[index]);
    }
}

bool ggml_cuda_reex_lut_test_launch(
        int device, enum ggml_cuda_reex_lut_test_op op,
        const float * input, float * output, size_t count) {
    if (input == nullptr || output == nullptr || count == 0 ||
        (op != GGML_CUDA_REEX_LUT_TEST_RECIPROCAL && op != GGML_CUDA_REEX_LUT_TEST_RSQRT)) {
        return false;
    }

    float * device_input = nullptr;
    float * device_output = nullptr;
    const size_t bytes = count * sizeof(float);
    if (cudaSetDevice(device) != cudaSuccess) {
        return false;
    }
    ggml_cuda_init_trigonometric_lut_REEX(device);
    if (cudaMalloc(&device_input, bytes) != cudaSuccess) {
        return false;
    }
    if (cudaMalloc(&device_output, bytes) != cudaSuccess) {
        cudaFree(device_input);
        return false;
    }

    bool ok = cudaMemcpy(device_input, input, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok) {
        const int block_size = 256;
        const int block_count = (int) ((count + block_size - 1) / block_size);
        ggml_cuda_reex_lut_test_kernel<<<block_count, block_size>>>(
            device_input, device_output, count, (int) op);
        ok = cudaGetLastError() == cudaSuccess;
    }
    if (ok) {
        ok = cudaMemcpy(output, device_output, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaFree(device_output);
    cudaFree(device_input);
    return ok;
}

#endif /* GGML_USE_REEX */
