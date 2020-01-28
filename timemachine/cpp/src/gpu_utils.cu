#include "gpu_utils.cuh"

cusparseStatus_t cusparseBsrmm(cusparseHandle_t         handle,
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
               int                      ldc) {
    return cusparseSbsrmm(
        handle,
        dirA,
        transA,
        transB,
        mb,
        n,
        kb,
        nnzb,
        alpha,
        descrA,
        bsrValA,
        bsrRowPtrA,
        bsrColIndA,
        blockDim,
        B,
        ldb,
        beta,
        C,
        ldc);

}

cusparseStatus_t cusparseBsrmm(cusparseHandle_t         handle,
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
               int                      ldc) {
    return cusparseDbsrmm(
        handle,
        dirA,
        transA,
        transB,
        mb,
        n,
        kb,
        nnzb,
        alpha,
        descrA,
        bsrValA,
        bsrRowPtrA,
        bsrColIndA,
        blockDim,
        B,
        ldb,
        beta,
        C,
        ldc);

}



cusparseStatus_t cusparseCsr2bsr(cusparseHandle_t         handle,
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
    int*                     bsrColIndC) {

    return cusparseScsr2bsr(
        handle,
        dir,
        m,
        n,
        descrA,
        csrValA,
        csrRowPtrA,
        csrColIndA,
        blockDim,
        descrC,
        bsrValC,
        bsrRowPtrC,
        bsrColIndC);

}

cusparseStatus_t cusparseCsr2bsr(cusparseHandle_t         handle,
    cusparseDirection_t      dir,
    int                      m,
    int                      n,
    const cusparseMatDescr_t descrA,
    const double*             csrValA,
    const int*               csrRowPtrA,
    const int*               csrColIndA,
    int                      blockDim,
    const cusparseMatDescr_t descrC,
    double*                   bsrValC,
    int*                     bsrRowPtrC,
    int*                     bsrColIndC) {

    return cusparseDcsr2bsr(
        handle,
        dir,
        m,
        n,
        descrA,
        csrValA,
        csrRowPtrA,
        csrColIndA,
        blockDim,
        descrC,
        bsrValC,
        bsrRowPtrC,
        bsrColIndC);

}


cusparseStatus_t cusparseCsrgemm2_bufferSizeExt(cusparseHandle_t         handle,
                                int                      m,
                                int                      n,
                                int                      k,
                                const float*             alpha,
                                const cusparseMatDescr_t descrA,
                                int                      nnzA,
                                const int*               csrRowPtrA,
                                const int*               csrColIndA,
                                const cusparseMatDescr_t descrB,
                                int                      nnzB,
                                const int*               csrRowPtrB,
                                const int*               csrColIndB,
                                const float*             beta,
                                const cusparseMatDescr_t descrD,
                                int                      nnzD,
                                const int*               csrRowPtrD,
                                const int*               csrColIndD,
                                csrgemm2Info_t           info,
                                size_t*                  pBufferSizeInBytes) {
    return cusparseScsrgemm2_bufferSizeExt(handle,
        m,
        n,
        k,
        alpha,
        descrA,
        nnzA,
        csrRowPtrA,
        csrColIndA,
        descrB,
        nnzB,
        csrRowPtrB,
        csrColIndB,
        beta,
        descrD,
        nnzD,
        csrRowPtrD,
        csrColIndD,
        info,
        pBufferSizeInBytes);

}

cusparseStatus_t cusparseCsrgemm2_bufferSizeExt(cusparseHandle_t         handle,
                                int                      m,
                                int                      n,
                                int                      k,
                                const double*            alpha,
                                const cusparseMatDescr_t descrA,
                                int                      nnzA,
                                const int*               csrRowPtrA,
                                const int*               csrColIndA,
                                const cusparseMatDescr_t descrB,
                                int                      nnzB,
                                const int*               csrRowPtrB,
                                const int*               csrColIndB,
                                const double*            beta,
                                const cusparseMatDescr_t descrD,
                                int                      nnzD,
                                const int*               csrRowPtrD,
                                const int*               csrColIndD,
                                csrgemm2Info_t           info,
                                size_t*                  pBufferSizeInBytes) {
    return cusparseDcsrgemm2_bufferSizeExt(handle,
        m,
        n,
        k,
        alpha,
        descrA,
        nnzA,
        csrRowPtrA,
        csrColIndA,
        descrB,
        nnzB,
        csrRowPtrB,
        csrColIndB,
        beta,
        descrD,
        nnzD,
        csrRowPtrD,
        csrColIndD,
        info,
        pBufferSizeInBytes);

}

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
    int*                     csrColIndC) {
    return cusparseScsrgemm(handle,
         transA,
         transB,
         m,
         n,
         k,
         descrA,
         nnzA,
         csrValA,
         csrRowPtrA,
         csrColIndA,
         descrB,
         nnzB,
         csrValB,
         csrRowPtrB,
         csrColIndB,
         descrC,
         csrValC,
         csrRowPtrC,
         csrColIndC);
}

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
    int*                     csrColIndC) {
    return cusparseDcsrgemm(handle,
         transA,
         transB,
         m,
         n,
         k,
         descrA,
         nnzA,
         csrValA,
         csrRowPtrA,
         csrColIndA,
         descrB,
         nnzB,
         csrValB,
         csrRowPtrB,
         csrColIndB,
         descrC,
         csrValC,
         csrRowPtrC,
         csrColIndC);
}

cusparseStatus_t cusparseDense2csr(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const float*             A,
                   int                      lda,
                   const int*               nnzPerRow,
                   float*                   csrValA,
                   int*                     csrRowPtrA,
                   int*                     csrColIndA) {
    return cusparseSdense2csr(handle,
        m,
        n,
        descrA,
        A,
        lda,
        nnzPerRow,
        csrValA,
        csrRowPtrA,
        csrColIndA);
}

cusparseStatus_t cusparseDense2csr(cusparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const cusparseMatDescr_t descrA,
                   const double*             A,
                   int                      lda,
                   const int*               nnzPerRow,
                   double*                   csrValA,
                   int*                     csrRowPtrA,
                   int*                     csrColIndA) {
    return cusparseDdense2csr(handle,
        m,
        n,
        descrA,
        A,
        lda,
        nnzPerRow,
        csrValA,
        csrRowPtrA,
        csrColIndA);
}

cusparseStatus_t cusparseNnz(cusparseHandle_t handle,
    cusparseDirection_t     dirA,
    int                      m,
    int                      n,
    const cusparseMatDescr_t descrA,
    const double*             A,
    int                      lda,
    int*                     nnzPerRowColumn,
    int*                     nnzTotalDevHostPtr) {

    return cusparseDnnz(handle,
             dirA,
             m,
             n,
             descrA,
             A,
             lda,
             nnzPerRowColumn,
             nnzTotalDevHostPtr);

}

cusparseStatus_t cusparseNnz(cusparseHandle_t handle,
    cusparseDirection_t     dirA,
    int                      m,
    int                      n,
    const cusparseMatDescr_t descrA,
    const float*             A,
    int                      lda,
    int*                     nnzPerRowColumn,
    int*                     nnzTotalDevHostPtr) {

    return cusparseSnnz(handle,
             dirA,
             m,
             n,
             descrA,
             A,
             lda,
             nnzPerRowColumn,
             nnzTotalDevHostPtr);

}

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
                           float           *C, int ldc) {

  return cublasSsymm(handle,
    side, uplo,
    m, n,
    alpha,
    A, lda,
    B, ldb, 
    beta,
    C, ldc);
}

cublasStatus_t templateSymm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           int m, int n,
                           const double           *alpha,
                           const double           *A, int lda,
                           const double           *B, int ldb,
                           const double           *beta,
                           double           *C, int ldc) {

  return cublasDsymm(handle,
    side, uplo,
    m, n,
    alpha,
    A, lda,
    B, ldb, 
    beta,
    C, ldc);
}

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

// #include <iostream> 
curandStatus_t templateCurandNormal(
    curandGenerator_t generator, 
    double *outputPtr, size_t n, 
    double mean, double stddev) {
    // std::cout << "N DOUBLE" << n << std::endl;
    return curandGenerateNormalDouble(generator, outputPtr, n, mean, stddev);
}
