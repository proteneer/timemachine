#include "k_indices.cuh"

// Takes a source and destination array. Assumes K <= N with values in the src are less than or equal
// to K. The value of the src is used as the indice and the value in the destination array. Allows combining
// a series of indices to get a unique set of values.
void __global__ k_unique_indices(
    const int N, // Number of values in src
    const int K, // Number of values in dest
    const unsigned int *__restrict__ src,
    unsigned int *__restrict__ dest) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    const unsigned int val = src[idx];
    if (val >= K) {
        return;
    }
    dest[val] = val;
}

// Any value that is >=N becomes the idx and any value that is an idx becomes N
void __global__ k_invert_indices(const int N, unsigned int *__restrict__ arr) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }

    arr[idx] = arr[idx] >= N ? idx : N;
}

void __global__ k_arange(const int N, unsigned int *__restrict__ arr) {
    const int atom_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom_idx >= N) {
        return;
    }
    arr[atom_idx] = atom_idx;
}
