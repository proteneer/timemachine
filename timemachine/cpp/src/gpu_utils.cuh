#pragma once

#include "curand.h"
#include "kernels/kernel_utils.cuh"
#include <cstdio>
#include <iostream>

// round_up_even is important to generating random numbers with cuRand as the generators only generate
// sets that are divisible by the dimension (typically 2) and will return error CURAND_STATUS_LENGTH_NOT_MULTIPLE.
// https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1gb94a31d5c165858c96b6c18b70644437
int round_up_even(int count);

curandStatus_t templateCurandNormal(curandGenerator_t generator, float *outputPtr, size_t n, float mean, float stddev);

curandStatus_t
templateCurandNormal(curandGenerator_t generator, double *outputPtr, size_t n, double mean, double stddev);

#define gpuErrchk(ans)                                                                                                 \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#define curandErrchk(ans)                                                                                              \
    { curandAssert((ans), __FILE__, __LINE__); }
inline void curandAssert(curandStatus_t code, const char *file, int line, bool abort = true) {
    if (code != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "curand failure, code: %d %s %d\n", code, file, line);
        if (abort)
            exit(code);
    }
}

/* cudaSafeMalloc is equivalent to gpuErrchk(cudaMalloc(...)) except that it prints a warning message if the
* allocation is greater than a GiB.
*/
#define cudaSafeMalloc(ptr, size)                                                                                      \
    ({                                                                                                                 \
        const int cudaSafeMalloc__line = __LINE__;                                                                     \
        if (size > (1 << 30)) {                                                                                        \
            fprintf(stderr, "cudaSafeMalloc: allocation larger than 1GiB %s %d\n", __FILE__, cudaSafeMalloc__line);    \
        }                                                                                                              \
        gpuAssert(cudaMalloc(ptr, size), __FILE__, cudaSafeMalloc__line, true);                                        \
    })

// safe is for use of gpuErrchk
template <typename T> T *gpuErrchkCudaMallocAndCopy(const T *host_array, int count) {
    T *device_array;
    cudaSafeMalloc(&device_array, count * sizeof(*host_array));
    gpuErrchk(cudaMemcpy(device_array, host_array, count * sizeof(*host_array), cudaMemcpyHostToDevice));
    return device_array;
}

template <typename T> void __global__ k_initialize_array(int count, T *array, T val) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    array[idx] = val;
}

template <typename T> void initializeArray(int count, T *array, T val) {

    int tpb = default_threads_per_block;
    int B = (count + tpb - 1) / tpb; // total number of blocks we need to process
    // Nothing to allocate
    if (count == 0) {
        return;
    }
    k_initialize_array<<<B, tpb, 0>>>(count, array, val);
    gpuErrchk(cudaPeekAtLastError());
}

float __device__ __forceinline__ rmul_rn(float a, float b) { return __fmul_rn(a, b); }

double __device__ __forceinline__ rmul_rn(double a, double b) { return __dmul_rn(a, b); }

float __device__ __forceinline__ radd_rn(float a, float b) { return __fadd_rn(a, b); }

double __device__ __forceinline__ radd_rn(double a, double b) { return __dadd_rn(a, b); }
