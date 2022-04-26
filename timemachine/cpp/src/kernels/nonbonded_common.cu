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

void __global__ k_arange(const int N, unsigned int *__restrict__ arr) {
    const int atom_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom_idx >= N) {
        return;
    }
    arr[atom_idx] = atom_idx;
}
