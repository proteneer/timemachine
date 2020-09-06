#pragma once

#include <cuda_runtime.h>

namespace timemachine {

// *Not* guaranteed to be thread-safe.
class Potential {

public: 

    virtual ~Potential() {};

    void execute_host(
        const int N,
        const int P,
        const double *h_x,
        const double *h_p,
        const double *h_box,
        const double lambda, // lambda
        unsigned long long *h_du_dx,
        double *h_du_dp,
        double *h_du_dl,
        double *h_u
    );

    virtual void execute_device(
        const int N,
        const int P,
        const double *d_x,
        const double *d_p,
        const double *d_box,
        const double lambda,
        unsigned long long *d_du_dx,
        double *d_du_dp,
        double *d_du_dl,
        double *d_u,
        cudaStream_t stream
    ) = 0;



    // void execute_lambda_jvp_host(
    //     const int N,
    //     const double *h_coords_primals,
    //     const double *h_coords_tangents,
    //     const double lambda_primal,
    //     const double lambda_tangent,
    //     double *h_out_coords_primals,
    //     double *h_out_coords_tangents
    // );

    // virtual void execute_lambda_inference_device(
    //     const int N,
    //     const double *d_coords_primals,
    //     const double lambda_primal,
    //     unsigned long long *d_out_coords_primals,
    //     double *d_out_lambda_primals,
    //     double *d_out_energy_primal,
    //     cudaStream_t stream
    // ) = 0;

    // virtual void execute_lambda_jvp_device(
    //     const int N,
    //     const double *d_coords_primals,
    //     const double *d_coords_tangents,
    //     const double lambda_primal,
    //     const double lambda_tangent,
    //     double *d_out_coords_primals,
    //     double *d_out_coords_tangents,
    //     cudaStream_t stream
    // ) = 0;



};

}
