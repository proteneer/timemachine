#pragma once

#include <cuda_runtime.h>

namespace timemachine {

class Gradient {

public: 

    virtual ~Gradient() {};

    void execute_lambda_inference_host(
        const int N,
        const double *h_coords_primals,
        const double lambda_primal,
        unsigned long long *h_out_coords_primals,
        double *h_out_lambda_primals,
        double *h_out_energy_primal
    );

    void execute_lambda_jvp_host(
        const int N,
        const double *h_coords_primals,
        const double *h_coords_tangents,
        const double lambda_primal,
        const double lambda_tangent,
        double *h_out_coords_primals,
        double *h_out_coords_tangents
    );

    virtual void execute_lambda_inference_device(
        const int N,
        const double *d_coords_primals,
        const double lambda_primal,
        unsigned long long *d_out_coords_primals,
        double *d_out_lambda_primals,
        double *d_out_energy_primal,
        cudaStream_t stream
    ) = 0;

    virtual void execute_lambda_jvp_device(
        const int N,
        const double *d_coords_primals,
        const double *d_coords_tangents,
        const double lambda_primal,
        const double lambda_tangent,
        double *d_out_coords_primals,
        double *d_out_coords_tangents,
        cudaStream_t stream
    ) = 0;



};

}
