#include "nonbonded_common.cuh"

void __global__
k_add_ull_to_ull(const int N, const unsigned long long *__restrict__ src, unsigned long long *__restrict__ dest) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.y;
    int stride_idx = blockIdx.y;

    if (idx >= N) {
        return;
    }

    dest[idx * stride + stride_idx] += src[idx * stride + stride_idx];
}
