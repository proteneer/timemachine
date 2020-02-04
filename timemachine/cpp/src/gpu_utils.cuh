#pragma once

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