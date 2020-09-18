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

// safe is for use of gpuErrchk
template<typename T>
T* gpuErrchkCudaMallocAndCopy(const T *host_array, int count) {
    T* device_array;
    std::cout << "foo" << std::endl;
    gpuErrchk(cudaMalloc(&device_array, count*sizeof(*host_array)));
    std::cout << "bar" << std::endl;
    gpuErrchk(cudaMemcpy(device_array, host_array, count*sizeof(*host_array), cudaMemcpyHostToDevice));
    std::cout << "zar" << std::endl;
    return device_array;
}