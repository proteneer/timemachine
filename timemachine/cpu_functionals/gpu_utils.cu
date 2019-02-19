#include "gpu_utils.cuh"


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
