#pragma once

#include <iostream>
#include <cstdio>
#include "curand.h"

curandStatus_t templateCurandNormal(
    curandGenerator_t generator, 
    float *outputPtr, size_t n, 
    float mean, float stddev);

curandStatus_t templateCurandNormal(
    curandGenerator_t generator, 
    double *outputPtr, size_t n, 
    double mean, double stddev);





#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define curandErrchk(ans) { curandAssert((ans), __FILE__, __LINE__); }
inline void curandAssert(curandStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CURAND_STATUS_SUCCESS) 
   {
      fprintf(stderr,"curand failure, code: %d %s %d\n", code, file, line);
      if (abort) exit(code);
   }
}

#define NVRTC_SAFE_CALL(x)                                        \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      std::cerr << "\nerror: " #x " failed with error "           \
                << nvrtcGetErrorString(result) << '\n';           \
      exit(1);                                                    \
    }                                                             \
  } while(0)

// safe is for use of gpuErrchk
template<typename T>
T* gpuErrchkCudaMallocAndCopy(const T *host_array, int count) {
    T* device_array;
    gpuErrchk(cudaMalloc(&device_array, count*sizeof(*host_array)));
    gpuErrchk(cudaMemcpy(device_array, host_array, count*sizeof(*host_array), cudaMemcpyHostToDevice));
    return device_array;
}

template<typename T>
void __global__ k_initialize_array(int count, T *array, T val) {
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= count) {
        return;
    }

    array[idx] = val;
}

template<typename T>
void initializeArray(int count, T *array, T val) {

    int tpb = 32;
    int B = (count+tpb-1)/tpb; // total number of blocks we need to process

    k_initialize_array<<<B, tpb, 0>>>(count, array, val);
    gpuErrchk(cudaPeekAtLastError());

}