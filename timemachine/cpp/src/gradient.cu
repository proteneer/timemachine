
#include <iostream>

#include "gradient.hpp"
#include "gpu_utils.cuh"
#include "surreal.cuh"

namespace jankmachine {

void Potential::execute_host(
    const int N,
    const double *h_x, // [N,3]
    const double *h_params, // [P,]
    const double *h_box, // [3, 3]
    const double lambda, // [1]
    unsigned long long *h_du_dx, // [N,3]
    double *h_du_dp, // [P]
    double *h_du_dl, //
    double *h_u) {

    double *d_x;
    double *d_box;

    const int D = 3;

    gpuErrchk(cudaMalloc(&d_x, N*D*sizeof(double)));
    gpuErrchk(cudaMemcpy(d_x, h_x, N*D*sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_box, D*D*sizeof(double)));
    gpuErrchk(cudaMemcpy(d_box, h_box, D*D*sizeof(double), cudaMemcpyHostToDevice));

    unsigned long long *d_du_dx; // du/dx

    double *d_du_dl; // du/dl
    double *d_u = nullptr; // u

    // very important that these are initialized to zero since the kernels themselves just accumulate
    gpuErrchk(cudaMalloc(&d_du_dx, N*D*sizeof(unsigned long long)));
    gpuErrchk(cudaMemset(d_du_dx, 0, N*D*sizeof(unsigned long long)));
    gpuErrchk(cudaMalloc(&d_du_dl, sizeof(double)));
    gpuErrchk(cudaMemset(d_du_dl, 0, sizeof(double)));
    gpuErrchk(cudaMalloc(&d_u, sizeof(double)));
    gpuErrchk(cudaMemset(d_u, 0, sizeof(double)));


    this->execute_device(
        N,
        d_x, 
        lambda,
        d_du_dx,
        d_du_dl,
        d_u,
        static_cast<cudaStream_t>(0)
    );

    gpuErrchk(cudaMemcpy(h_du_dx, d_du_dx, N*D*sizeof(*h_du_dx), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_du_dx));
    gpuErrchk(cudaMemcpy(h_du_dl, d_du_dl, sizeof(*h_du_dl), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_du_dl));
    gpuErrchk(cudaMemcpy(h_u, d_u, sizeof(*h_u), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_u));
    gpuErrchk(cudaFree(d_coords));
    gpuErrchk(cudaFree(d_box));

};

// void Gradient::execute_lambda_jvp_host(
//     const int N,
//     const double *h_in_coords_primals,
//     const double *h_in_coords_tangents,
//     const double lambda_primal,
//     const double lambda_tangent,
//     double *h_out_coords_primals,
//     double *h_out_coords_tangents) {

//     double *d_in_coords_primals;
//     double *d_in_params_primals;

//     const int D = 3;

//     gpuErrchk(cudaMalloc(&d_in_coords_primals, N*D*sizeof(double)));
//     gpuErrchk(cudaMemcpy(d_in_coords_primals, h_in_coords_primals, N*D*sizeof(double), cudaMemcpyHostToDevice));

//     double *d_in_coords_tangents; // x tangent
//     double *d_out_coords_primals; // du/dx
//     double *d_out_coords_tangents; // du/dx

//     gpuErrchk(cudaMalloc(&d_in_coords_tangents, N*D*sizeof(double)));
//     gpuErrchk(cudaMemcpy(d_in_coords_tangents, h_in_coords_tangents, N*D*sizeof(double), cudaMemcpyHostToDevice));

//     gpuErrchk(cudaMalloc(&d_out_coords_tangents, N*D*sizeof(double)));
//     gpuErrchk(cudaMemset(d_out_coords_tangents, 0, N*D*sizeof(double)));

//     gpuErrchk(cudaMalloc(&d_out_coords_primals, N*D*sizeof(double)));
//     gpuErrchk(cudaMemset(d_out_coords_primals, 0, N*D*sizeof(double)));

//     this->execute_lambda_jvp_device(
//         N,
//         d_in_coords_primals, 
//         d_in_coords_tangents,
//         lambda_primal,
//         lambda_tangent,
//         d_out_coords_primals,
//         d_out_coords_tangents,
//         static_cast<cudaStream_t>(0)
//     );

//     gpuErrchk(cudaMemcpy(h_out_coords_tangents, d_out_coords_tangents, N*D*sizeof(double), cudaMemcpyDeviceToHost));
//     gpuErrchk(cudaFree(d_out_coords_tangents));

//     gpuErrchk(cudaMemcpy(h_out_coords_primals, d_out_coords_primals, N*D*sizeof(double), cudaMemcpyDeviceToHost));
//     gpuErrchk(cudaFree(d_out_coords_primals));

//     gpuErrchk(cudaFree(d_in_coords_primals));
//     gpuErrchk(cudaFree(d_in_coords_tangents));

// };

}

