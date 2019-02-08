#pragma once

#include "cublas_v2.h"

// #include <ctime>
#include <vector>
#include <stdexcept>
#include <cstdio>

namespace timemachine {

// Integrator owns *everything*. This design basically
// goes against the design of almost every text-book pattern.
// We store all memory intensive elements in here to make it easy
// to keep track of pointers and estimate total ram use.    
template<typename NumericType>
class Integrator {

private:

    cublasHandle_t cb_handle_;

    const int W_;
    const int N_;
    const int P_;

    size_t step_;

    // GPU device buffers
    NumericType *d_dxdp_t_;
    NumericType *d_total_buffer_;
    NumericType *d_converged_buffer_;
    NumericType *d_coeff_bs_;

    NumericType *d_hessians_;
    NumericType *d_mixed_partials_;

    NumericType coeff_a_;

    void reduce_buffers(const NumericType *d_Dx_t, int window_k);

    void hessian_vector_product(
        const NumericType *d_A,
        NumericType *d_B,
        NumericType *d_C);

public:

    std::vector<NumericType> get_dxdp() const;

    NumericType* get_device_hessians() {
        return d_hessians_;
    };

    NumericType* get_device_mixed_partials() {
        return d_mixed_partials_;
    };

    Integrator(
        NumericType coeff_a,
        int W,
        int N,
        int P,
        const std::vector<NumericType> &coeff_bs);

    ~Integrator();

    void step_cpu(
        const NumericType *h_hessians,
        const NumericType *h_mixed_partials);

    void step_gpu(
        const NumericType *d_hessians,
        NumericType *d_mixed_partials);


};

}

