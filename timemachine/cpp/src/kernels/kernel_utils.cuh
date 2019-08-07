#pragma once

#include <cstdio>
#define WARP_SIZE 32

#define HESS_3N3N(i,j,N,di,dj) (di*N*3*N + i*3*N + dj*N + j)
#define HESS_N3N3(i,j,N,di,dj) (i*3*N*3 + di*N*3 + j*3 + dj)
#define HESS_IDX HESS_N3N3

#define HESS_NDND(i,j,N,di,dj,D) (i*D*N*D + di*N*D + j*D + dj)
#define HESS_IDX_ND HESS_NDND

#define ONE_4PI_EPS0 138.935456

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


inline __device__ float gpuSqrt(float arg) {
  return sqrtf(arg);
}

inline __device__ double gpuSqrt(double arg) {
  return sqrt(arg);
}
