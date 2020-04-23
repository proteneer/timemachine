#pragma once

#include <cuda_runtime.h>

namespace timemachine {

template <int D>
class Gradient {

public: 

    virtual ~Gradient() {};

    /*
    Take in pointers to host memory.
    */
    void execute_host(
        const int N,
        const int P,
        const double *h_in_coords,
        const double *h_in_coords_tangents,
        const double *h_in_params,
        unsigned long long *h_out_coords,
        double *h_out_coords_tangents,
        double *h_out_params_tangents
    );

    void execute_lambda_host(
        const int N,
        const int P,
        const double *h_in_coords_primals,
        const double *h_in_coords_tangents, // inference
        const double *h_in_params_primals,
        const double lambda_primal,
        const double lambda_tangent, // inference
        unsigned long long *h_out_coords_primals, // inference
        double *h_out_lambda_primals, // inference

        double *h_out_coords_tangents, // jvp
        double *h_out_params_tangents // jvp
    );

    virtual void execute_device(
        const int N,
        const int P,
        const double *d_coords,
        const double *d_coords_tangents,
        const double *d_params,
        unsigned long long *d_out_coords,
        double *d_out_coords_tangents,
        double *d_out_params_tangents,
        cudaStream_t stream
    ) = 0;

    virtual void execute_lambda_device(
        const int N,
        const int P,
        const double *d_coords_primals,
        const double *d_coords_tangents,
        const double *d_params_primals,
        const double lambda_primal,
        const double lambda_tangent,
        unsigned long long *d_out_coords_primals,
        double *d_out_lambda_primals,
        double *d_out_coords_tangents,
        double *d_out_params_tangents,
        cudaStream_t stream
    ) = 0;

};

}
