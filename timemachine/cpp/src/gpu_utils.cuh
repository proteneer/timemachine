#pragma once

#include <cstdio>
#include "cublas_v2.h"
#include "cusparse_v2.h"
#include "curand.h"

cusparseStatus_t
cusparseBsrmm(cusparseHandle_t         handle,
               cusparseDirection_t      dirA,
               cusparseOperation_t      transA,
               cusparseOperation_t      transB,
               int                      mb,
               int                      n,
               int                      kb,
               int                      nnzb,
               const float*             alpha,
               const cusparseMatDescr_t descrA,
               const float*             bsrValA,
               const int*               bsrRowPtrA,
               const int*               bsrColIndA,
               int                      blockDim,
               const float*             B,
               int                      ldb,
               const float*             beta,
               float*                   C,
               int                      ldc);

cusparseStatus_t
cusparseBsrmm(cusparseHandle_t         handle,
               cusparseDirection_t      dirA,
               cusparseOperation_t      transA,
               cusparseOperation_t      transB,
               int                      mb,
               int                      n,
               int                      kb,
               int                      nnzb,
               const double*            alpha,
               const cusparseMatDescr_t descrA,
               const double*            bsrValA,
               const int*               bsrRowPtrA,
               const int*               bsrColIndA,
               int                      blockDim,
               const double*            B,
               int                      ldb,
               const double*            beta,
               double*                  C,
               int                      ldc);


cusparseStatus_t
cusparseCsr2bsr(cusparseHandle_t         handle,
                 cusparseDirection_t      dir,
                 int                      m,
                 int                      n,
                 const cusparseMatDescr_t descrA,
                 const float*             csrValA,
                 const int*               csrRowPtrA,
                 const int*               csrColIndA,
                 int                      blockDim,
                 const cusparseMatDescr_t descrC,
                 float*                   bsrValC,
                 int*                     bsrRowPtrC,
                 int*                     bsrColIndC);

cusparseStatus_t
cusparseCsr2bsr(cusparseHandle_t         handle,
                 cusparseDirection_t      dir,
                 int                      m,
                 int                      n,
                 const cusparseMatDescr_t descrA,
                 const double*            csrValA,
                 const int*               csrRowPtrA,
                 const int*               csrColIndA,
                 int                      blockDim,
                 const cusparseMatDescr_t descrC,
                 double*                  bsrValC,
                 int*                     bsrRowPtrC,
                 int*                     bsrColIndC);

cusparseStatus_t
cusparseCsrgemm(cusparseHandle_t        handle,
    cusparseOperation_t      transA,
    cusparseOperation_t      transB,
    int                      m,
    int                      n,
    int                      k,
    const cusparseMatDescr_t descrA,
    int                      nnzA,
    const float*             csrValA,
    const int*               csrRowPtrA,
    const int*               csrColIndA,
    const cusparseMatDescr_t descrB,
    int                      nnzB,
    const float*             csrValB,
    const int*               csrRowPtrB,
    const int*               csrColIndB,
    const cusparseMatDescr_t descrC,
    float*                   csrValC,
    const int*               csrRowPtrC,
    int*                     csrColIndC);

cusparseStatus_t
cusparseCsrgemm(cusparseHandle_t        handle,
    cusparseOperation_t      transA,
    cusparseOperation_t      transB,
    int                      m,
    int                      n,
    int                      k,
    const cusparseMatDescr_t descrA,
    int                      nnzA,
    const double*            csrValA,
    const int*               csrRowPtrA,
    const int*               csrColIndA,
    const cusparseMatDescr_t descrB,
    int                      nnzB,
    const double*            csrValB,
    const int*               csrRowPtrB,
    const int*               csrColIndB,
    const cusparseMatDescr_t descrC,
    double*                  csrValC,
    const int*               csrRowPtrC,
    int*                     csrColIndC);


cusparseStatus_t cusparseDense2csr(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const float*             A,
                   int                      lda,
                   const int*               nnzPerRow,
                   float*                   csrValA,
                   int*                     csrRowPtrA,
                   int*                     csrColIndA);

cusparseStatus_t cusparseDense2csr(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const double*             A,
                   int                      lda,
                   const int*               nnzPerRow,
                   double*                   csrValA,
                   int*                     csrRowPtrA,
                   int*                     csrColIndA);

cusparseStatus_t cusparseNnz(cusparseHandle_t handle,
    cusparseDirection_t     dirA,
    int                      m,
    int                      n,
    const cusparseMatDescr_t descrA,
    const double*             A,
    int                      lda,
    int*                     nnzPerRowColumn,
    int*                     nnzTotalDevHostPtr);

cusparseStatus_t cusparseNnz(cusparseHandle_t handle,
    cusparseDirection_t     dirA,
    int                      m,
    int                      n,
    const cusparseMatDescr_t descrA,
    const float*             A,
    int                      lda,
    int*                     nnzPerRowColumn,
    int*                     nnzTotalDevHostPtr);


cublasStatus_t templateSymm(cublasHandle_t handle,
  cublasSideMode_t side, cublasFillMode_t uplo,
  int m, int n,
  const float           *alpha,
  const float           *A, int lda,
  const float           *B, int ldb,
  const float           *beta,
  float           *C, int ldc);

cublasStatus_t templateSymm(cublasHandle_t handle,
  cublasSideMode_t side, cublasFillMode_t uplo,
  int m, int n,
  const double           *alpha,
  const double           *A, int lda,
  const double           *B, int ldb,
  const double           *beta,
  double           *C, int ldc);

cublasStatus_t templateGemm(cublasHandle_t handle,
   cublasOperation_t transa, cublasOperation_t transb,
   int m, int n, int k,
   const float           *alpha,
   const float           *A, int lda,
   const float           *B, int ldb,
   const float           *beta,
   float           *C, int ldc);

cublasStatus_t templateGemm(cublasHandle_t handle,
   cublasOperation_t transa, cublasOperation_t transb,
   int m, int n, int k,
   const double           *alpha,
   const double           *A, int lda,
   const double           *B, int ldb,
   const double           *beta,
   double           *C, int ldc);

curandStatus_t templateCurandNormal(
    curandGenerator_t generator, 
    float *outputPtr, size_t n, 
    float mean, float stddev);

curandStatus_t templateCurandNormal(
    curandGenerator_t generator, 
    double *outputPtr, size_t n, 
    double mean, double stddev);


#define cusparseErrchk(ans) { cusparseAssert((ans), __FILE__, __LINE__); }
inline void cusparseAssert(cusparseStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUSPARSE_STATUS_SUCCESS) 
   {
      fprintf(stderr,"cublas failure, code: %d %s %d\n", code, file, line);
      if (abort) exit(code);
   }
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