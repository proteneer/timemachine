#pragma once
#include "cublas_v2.h"

// #include <ctime>
#include <stdexcept>
#include <cstdio>

#define cublasErrchk(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUBLAS_STATUS_SUCCESS) 
   {
      fprintf(stderr,"cublas failure, code: %d %s %d\n", code, file, line);
      if (abort) exit(code);
   }
}

namespace timemachine {

template<typename NumericType>
class Integrator {

private:

    cublasHandle_t cb_handle_;


public:

    Integrator() {
        cublasCreate(&cb_handle_);
    }

    ~Integrator() {
        cublasDestroy(cb_handle_);
    }


    void hessian_vector_product(
        NumericType *d_A,
        NumericType *d_B,
        NumericType *d_C,
        int N3,
        int P) {

        float alpha = 1.0;
        float beta  = 1.0;
     
        // replace with DGEMM later
        cublasErrchk(cublasSgemm(cb_handle_,
                    CUBLAS_OP_N, CUBLAS_OP_N, // whether or not we transpose A
                    N3, P, N3,
                    &alpha,
                    d_A, N3,
                    d_B, N3,
                    &beta,
                    d_C, N3));
    }

};

}

