#pragma once

cublasStatus_t templateGemm(cublasHandle_t handle,
   cublasOperation_t transa, cublasOperation_t transb,
   int m, int n, int k,
   const float           *alpha,
   const float           *A, int lda,
   const float           *B, int ldb,
   const float           *beta,
   float           *C, int ldc) {
   return cublasSgemm(handle,
   transa, transb,
   m, n, k,
   alpha,
   A, lda,
   B, ldb,
   beta,
   C, ldc);
}

cublasStatus_t templateGemm(cublasHandle_t handle,
   cublasOperation_t transa, cublasOperation_t transb,
   int m, int n, int k,
   const double           *alpha,
   const double           *A, int lda,
   const double           *B, int ldb,
   const double           *beta,
   double           *C, int ldc) {
   return cublasDgemm(handle,
   transa, transb,
   m, n, k,
   alpha,
   A, lda,
   B, ldb,
   beta,
   C, ldc);
}

curandStatus_t templateCurandNormal(
    curandGenerator_t generator, 
    float *outputPtr, size_t n, 
    float mean, float stddev) {
    return curandGenerateNormal(generator, outputPtr, n, mean, stddev);
}

curandStatus_t templateCurandNormal(
    curandGenerator_t generator, 
    double *outputPtr, size_t n, 
    double mean, double stddev) {
    return curandGenerateNormalDouble(generator, outputPtr, n, mean, stddev);
}


#define cublasErrchk(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUBLAS_STATUS_SUCCESS) 
   {
      fprintf(stderr,"cublas failure, code: %d %s %d\n", code, file, line);
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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}