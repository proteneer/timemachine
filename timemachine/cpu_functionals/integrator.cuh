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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template<typename NumericType>
__global__ void reduce_total(
    NumericType coeff_a,
    NumericType *coeff_bs,
    NumericType *buffer,
    size_t k, // starting window slot
    size_t W, // number of windows
    size_t PN3 // PN3
    ) {

    NumericType prefactor = 0.0;
    NumericType a_n = 1.0;
    NumericType accum = 0.0;

    // divide up in multiple grids
    for(size_t w=k; w < W; w++) {

        size_t w_pn3_idx = w*PN3 + blockIdx.x*blockDim.x + threadIdx.x;

        // size_t pn3_idx = blockIdx.x*blockDim.x + threadIdx.x;
        if(w_pn3_idx >= W * PN3) {
            // happens on the last warp
            return;
        }

 
        prefactor += a_n;
        a_n *= coeff_a;
        accum += prefactor*buffer[w_pn3_idx];
    }
    size_t pn3_idx = k*W*PN3 + blockIdx.x*blockDim.x + threadIdx.x;
    // coeff_b's can be optimized into smaller chunks.
    buffer[pn3_idx] = coeff_bs[pn3_idx] * accum;
    // overwrite into w slot
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

    // void reduce_total_derivatives(
    //     NumericType *d_A,
    //     size_t W,
    //     size_t B) {


    // }

};

}

