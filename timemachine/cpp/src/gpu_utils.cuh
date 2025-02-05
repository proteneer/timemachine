#pragma once

#include "curand.h"
#include "curand_kernel.h"
#include "exceptions.hpp"
#include "fixed_point.hpp"
#include "kernels/kernel_utils.cuh"
#include <iostream>

namespace timemachine {

// round_up_even is important to generating random numbers with cuRand if generating Normal noise as the normal generators only generate
// sets that are divisible by the dimension (typically 2) and will return error CURAND_STATUS_LENGTH_NOT_MULTIPLE.
// https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1gb94a31d5c165858c96b6c18b70644437
int round_up_even(int count);

curandStatus_t templateCurandNormal(curandGenerator_t generator, float *outputPtr, size_t n, float mean, float stddev);

curandStatus_t
templateCurandNormal(curandGenerator_t generator, double *outputPtr, size_t n, double mean, double stddev);

curandStatus_t templateCurandUniform(curandGenerator_t generator, float *outputPtr, size_t n);

curandStatus_t templateCurandUniform(curandGenerator_t generator, double *outputPtr, size_t n);

#define gpuErrchk(ans)                                                                                                 \
    {                                                                                                                  \
        gpuAssert((ans), __FILE__, __LINE__);                                                                          \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            // If the GPU is invalid or missing for some reason, raise an exception so we can handle that
            // Error codes can be found here: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038
            switch (code) {
            case cudaErrorInvalidDevice:
            case cudaErrorInsufficientDriver:
            case cudaErrorNoDevice:
            case cudaErrorStartupFailure:
            case cudaErrorInvalidPtx:
            case cudaErrorUnsupportedPtxVersion:
            case cudaErrorDevicesUnavailable:
            case cudaErrorUnknown:
                throw InvalidHardware(code);
            default:
                break;
            }
            exit(code);
        }
    }
}

#define curandErrchk(ans)                                                                                              \
    {                                                                                                                  \
        curandAssert((ans), __FILE__, __LINE__);                                                                       \
    }
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

// k_initialize_curand_states initializes an array of curandState_t objects such that each object
// uses the seed provided + the index in the array. Offsets and sequences always set to 0
void __global__ k_initialize_curand_states(const int count, const int seed, curandState_t *states);

template <typename T> void __global__ k_initialize_array(int count, T *array, T val) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    array[idx] = val;
}

template <typename T> void initializeArray(int count, T *array, T val) {

    int tpb = DEFAULT_THREADS_PER_BLOCK;
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

float __device__ __forceinline__ rsub_rn(float a, float b) { return __fsub_rn(a, b); }

double __device__ __forceinline__ rsub_rn(double a, double b) { return __dsub_rn(a, b); }

// If this were not int128s, we could do __shfl_down_sync, but only supports up to 64 values
// For more details reference the PDF in the docstring for k_accumulate_energy
template <unsigned int THREADS_PER_BLOCK>
__device__ __forceinline__ void energy_warp_reduce(volatile __int128 *shared_data, int thread_idx) {
    // Repeatedly sum up the two halfs of the shared data
#pragma unroll 6
    for (int i = 32; i > 0; i /= 2) {
        if (thread_idx < i && thread_idx + i < THREADS_PER_BLOCK) {
            shared_data[thread_idx] = shared_data[thread_idx] + shared_data[thread_idx + i];
        }
        __syncwarp();
    }
}

// block_energy_reduce does a parallel reduction over shared memory within a block. The THREADS_PER_BLOCK
// template is used to handle the cases where the number of threads is greater than a warp and sums half of the
// values repeatedly until the final warp level reduction which stores the final accumulated value in `shared_data[0]`.
// For more details reference the PDF in the docstring for k_accumulate_energy
template <unsigned int THREADS_PER_BLOCK>
__device__ __forceinline__ void block_energy_reduce(
    volatile __int128 *shared_data, // [THREADS_PER_BLOCK]
    int tid) {
    static_assert(THREADS_PER_BLOCK <= 1024 && (THREADS_PER_BLOCK & (THREADS_PER_BLOCK - 1)) == 0);

    // For larger blocks larger than a warp
    if (THREADS_PER_BLOCK >= 1024) {
        if (tid < 512 && tid + 512 < THREADS_PER_BLOCK) {
            shared_data[tid] += shared_data[tid + 512];
        }
        __syncthreads();
    }
    if (THREADS_PER_BLOCK >= 512) {
        if (tid < 256 && tid + 256 < THREADS_PER_BLOCK) {
            shared_data[tid] += shared_data[tid + 256];
        }
        __syncthreads();
    }
    if (THREADS_PER_BLOCK >= 256) {
        if (tid < 128 && tid + 128 < THREADS_PER_BLOCK) {
            shared_data[tid] += shared_data[tid + 128];
        }
        __syncthreads();
    }
    if (THREADS_PER_BLOCK >= 128) {
        if (tid < 64 && tid + 64 < THREADS_PER_BLOCK) {
            shared_data[tid] += shared_data[tid + 64];
        }
        __syncthreads();
    }

    if (tid < WARP_SIZE) {
        energy_warp_reduce<THREADS_PER_BLOCK>(shared_data, tid);
    }
}

// Taken originally from https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
// but modified to account for post Cuda 9.0 which no longer implicitly syncs warps and the
// fact that we don't need to launch recursive blocks as we can simply reduce the number of energies
// that are accumulated.
template <unsigned int BLOCK_SIZE>
void __global__ k_accumulate_energy(
    int N,
    const __int128 *__restrict__ input_buffer, // [N]
    __int128 *__restrict__ u_buffer            // [1]
) {

    __shared__ __int128 shared_mem[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    if (tid >= BLOCK_SIZE) {
        return;
    }

    unsigned int i = tid;
    const unsigned int stride = BLOCK_SIZE * 2;
    shared_mem[tid] = 0;
    while (i < N) {
        // Only add two values if the input buffer has enough input
        shared_mem[tid] += input_buffer[i] + (i + BLOCK_SIZE < N ? input_buffer[i + BLOCK_SIZE] : 0);
        i += stride;
    }
    __syncthreads();

    block_energy_reduce<BLOCK_SIZE>(shared_mem, threadIdx.x);
    if (tid == 0) {
        // This could be a race condition if multiple `k_accumulate_energy` are running
        // on the same u_buffer
        u_buffer[0] = shared_mem[0];
    }
}

} // namespace timemachine
