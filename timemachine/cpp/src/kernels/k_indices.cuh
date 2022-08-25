#pragma once
#include <cub/cub.cuh>

// Struct to as a CUB <= operation
struct LessThan {
    int compare;
    CUB_RUNTIME_FUNCTION __device__ __forceinline__ explicit LessThan(int compare) : compare(compare) {}
    CUB_RUNTIME_FUNCTION __device__ __forceinline__ bool operator()(const int &a) const { return (a < compare); }
};

// Takes a source and destination array. Assumes K <= N with values in the src are less than or equal
// to K. The value of the src is used as the indice and the value in the destination array. Allows combining
// a series of indices to get a unique set of values.
void __global__ k_unique_indices(
    const int N, // Number of values in src
    const int K, // Number of values in dest
    const unsigned int *__restrict__ src,
    unsigned int *__restrict__ dest);

// Any value that is >=N becomes the idx and any value that is an idx becomes N. Assumes
// that the array is made up of indice values that correspond to their index in the array,
// otherwise the inversion may contain values that were in the input.
void __global__ k_invert_indices(const int N, unsigned int *__restrict__ arr);

void __global__ k_arange(const int N, unsigned int *__restrict__ arr);
