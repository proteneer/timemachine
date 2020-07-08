
#include <iostream>

#include "gradient.hpp"
#include "gpu_utils.cuh"
#include "surreal.cuh"

namespace timemachine {

void Gradient::execute_lambda_inference_host(
    const int N,
    // const int P,
    const double *h_in_coords_primals,
    // const double *h_in_params_primals,
    const double lambda_primal,
    unsigned long long *h_out_coords_primals,
    double *h_out_lambda_primals,
    double *h_out_energy_primal) {

    double *d_in_coords_primals;
    // double *d_in_params_primals;

    const int D = 3;

    gpuErrchk(cudaMalloc(&d_in_coords_primals, N*D*sizeof(double)));
    // gpuErrchk(cudaMalloc(&d_in_params_primals, P*sizeof(double)));
    gpuErrchk(cudaMemcpy(d_in_coords_primals, h_in_coords_primals, N*D*sizeof(double), cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(d_in_params_primals, h_in_params_primals, P*sizeof(double), cudaMemcpyHostToDevice));

    unsigned long long *d_out_coords_primals; // du/dx
    double *d_out_lambda_primals; // du/dl
    double *d_out_energy = nullptr; // u

    // very important that these are initialized to zero since the kernels themselves just accumulate
    gpuErrchk(cudaMalloc(&d_out_coords_primals, N*D*sizeof(unsigned long long)));
    gpuErrchk(cudaMemset(d_out_coords_primals, 0, N*D*sizeof(unsigned long long)));
    gpuErrchk(cudaMalloc(&d_out_lambda_primals, sizeof(double)));
    gpuErrchk(cudaMemset(d_out_lambda_primals, 0, sizeof(double)));
    gpuErrchk(cudaMalloc(&d_out_energy, sizeof(double)));
    gpuErrchk(cudaMemset(d_out_energy, 0, sizeof(double)));


    this->execute_lambda_inference_device(
        N,
        // P,
        d_in_coords_primals, 
        // d_in_params_primals,
        lambda_primal,
        d_out_coords_primals,
        d_out_lambda_primals,
        d_out_energy,
        static_cast<cudaStream_t>(0)
    );

    gpuErrchk(cudaMemcpy(h_out_coords_primals, d_out_coords_primals, N*D*sizeof(*h_out_coords_primals), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_out_coords_primals));
    gpuErrchk(cudaMemcpy(h_out_lambda_primals, d_out_lambda_primals, sizeof(*h_out_lambda_primals), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_out_lambda_primals));
    gpuErrchk(cudaMemcpy(h_out_energy_primal, d_out_energy, sizeof(*h_out_energy_primal), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_out_energy));

    gpuErrchk(cudaFree(d_in_coords_primals));
    // gpuErrchk(cudaFree(d_in_params_primals));

};

void Gradient::execute_lambda_jvp_host(
    const int N,
    // const int P,
    const double *h_in_coords_primals,
    const double *h_in_coords_tangents,
    // const double *h_in_params_primals,
    const double lambda_primal,
    const double lambda_tangent,
    double *h_out_coords_primals,
    double *h_out_coords_tangents
    // double *h_out_params_primals,
    // double *h_out_params_tangents
    ) {

    double *d_in_coords_primals;
    double *d_in_params_primals;

    const int D = 3;

    gpuErrchk(cudaMalloc(&d_in_coords_primals, N*D*sizeof(double)));
    // gpuErrchk(cudaMalloc(&d_in_params_primals, P*sizeof(double)));
    gpuErrchk(cudaMemcpy(d_in_coords_primals, h_in_coords_primals, N*D*sizeof(double), cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(d_in_params_primals, h_in_params_primals, P*sizeof(double), cudaMemcpyHostToDevice));

    double *d_in_coords_tangents; // x tangent
    double *d_out_coords_primals; // du/dx
    double *d_out_coords_tangents; // du/dx
    // double *d_out_params_primals; // du/dp
    // double *d_out_params_tangents; // du/dp

    gpuErrchk(cudaMalloc(&d_in_coords_tangents, N*D*sizeof(double)));
    gpuErrchk(cudaMemcpy(d_in_coords_tangents, h_in_coords_tangents, N*D*sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_out_coords_tangents, N*D*sizeof(double)));
    // gpuErrchk(cudaMalloc(&d_out_params_tangents, P*sizeof(double)));
    gpuErrchk(cudaMemset(d_out_coords_tangents, 0, N*D*sizeof(double)));
    // gpuErrchk(cudaMemset(d_out_params_tangents, 0, P*sizeof(double)));

    gpuErrchk(cudaMalloc(&d_out_coords_primals, N*D*sizeof(double)));
    // gpuErrchk(cudaMalloc(&d_out_params_primals, P*sizeof(double)));
    gpuErrchk(cudaMemset(d_out_coords_primals, 0, N*D*sizeof(double)));
    // gpuErrchk(cudaMemset(d_out_params_primals, 0, P*sizeof(double)));

    this->execute_lambda_jvp_device(
        N,
        // P,
        d_in_coords_primals, 
        d_in_coords_tangents,
        // d_in_params_primals,
        lambda_primal,
        lambda_tangent,
        d_out_coords_primals,
        d_out_coords_tangents,
        // d_out_params_primals,
        // d_out_params_tangents,
        static_cast<cudaStream_t>(0)
    );

    gpuErrchk(cudaMemcpy(h_out_coords_tangents, d_out_coords_tangents, N*D*sizeof(double), cudaMemcpyDeviceToHost));
    // gpuErrchk(cudaMemcpy(h_out_params_tangents, d_out_params_tangents, P*sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_out_coords_tangents));
    // gpuErrchk(cudaFree(d_out_params_tangents));

    gpuErrchk(cudaMemcpy(h_out_coords_primals, d_out_coords_primals, N*D*sizeof(double), cudaMemcpyDeviceToHost));
    // gpuErrchk(cudaMemcpy(h_out_params_primals, d_out_params_primals, P*sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_out_coords_primals));
    // gpuErrchk(cudaFree(d_out_params_primals));

    gpuErrchk(cudaFree(d_in_coords_primals));
    gpuErrchk(cudaFree(d_in_coords_tangents));
    // gpuErrchk(cudaFree(d_in_params_primals));

};

}

