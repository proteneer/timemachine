#pragma once

#include "cublas_v2.h"
#include "curand.h"

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
    curandGenerator_t  cr_rng_;

    const int W_;
    const int N_;
    const int P_;

    size_t step_;


    NumericType dt_;

    // GPU device buffers
    NumericType *d_x_t_; // geometries
    NumericType *d_v_t_; // velocities
    NumericType *d_dxdp_t_; // derivatives of geometry wrt parameters
    NumericType *d_total_buffer_; // total derivatives
    NumericType *d_converged_buffer_;


    // should these be owned by the context instead? we want them to be explicit
    NumericType *d_energy_;
    NumericType *d_grads_;
    NumericType *d_hessians_;
    NumericType *d_mixed_partials_;

    NumericType *d_rng_buffer_;

    NumericType coeff_a_;
    NumericType *d_coeff_bs_;
    NumericType *d_coeff_cs_;

    // void reduce_velocities()
    // void reduce_total_derivatives(const NumericType *d_Dx_t, int window_k);

    void hessian_vector_product(
        const NumericType *d_A,
        NumericType *d_B,
        NumericType *d_C);

public:

    void reset();

    NumericType* get_device_energy() {
      return d_energy_;
    }

    // these *HAVE* to be refactored out to be out of this class.
    NumericType* get_device_coords() {
        return d_x_t_;
    };

    // these *HAVE* to be refactored out to be out of this class.
    NumericType* get_device_grads() {
        return d_grads_;
    };

    // these *HAVE* to be refactored out to be out of this class.
    NumericType* get_device_hessians() {
        return d_hessians_;
    };

    NumericType* get_device_mixed_partials() {
        return d_mixed_partials_;
    };

    int num_params() const {
      return P_;
    }

    int num_atoms() const {
      return N_;
    }

    std::vector<NumericType> get_dxdp() const;

    std::vector<NumericType> get_noise() const;

    std::vector<NumericType> get_coordinates() const;

    std::vector<NumericType> get_velocities() const;

    void set_coordinates(std::vector<NumericType>);

    void set_velocities(std::vector<NumericType>);

    Integrator(
        NumericType dt,
        int W,
        int N,
        int P,
        const NumericType coeff_a,
        const std::vector<NumericType> &coeff_bs,
        const std::vector<NumericType> &coeff_cs);

    ~Integrator();

    void step_cpu(
        const NumericType *h_grads,
        const NumericType *h_hessians,
        const NumericType *h_mixed_partials);

    void step_gpu(
        const NumericType *d_grads,
        const NumericType *d_hessians,
        NumericType *d_mixed_partials);


};

}

