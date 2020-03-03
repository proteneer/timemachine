
#include "gradient.hpp"
#include "kernel_utils.cuh"
#include "surreal.cuh"

namespace timemachine {

template<int D>
void Gradient<D>::execute_host(
    const int N,
    const int P,
    const double *h_in_coords,
    const double *h_in_coords_tangents,
    const double *h_in_params,
    unsigned long long *h_out_coords,
    double *h_out_coords_tangents,
    double *h_out_params_tangents) {

    double *d_in_coords;
    double *d_in_params;

    gpuErrchk(cudaMalloc(&d_in_coords, N*D*sizeof(double)));
    gpuErrchk(cudaMalloc(&d_in_params, P*sizeof(double)));
    gpuErrchk(cudaMemcpy(d_in_coords, h_in_coords, N*D*sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_in_params, h_in_params, P*sizeof(double), cudaMemcpyHostToDevice));

    unsigned long long *d_out_coords;

    // very important that we initialize these
    double *d_in_coords_tangents = nullptr;
    double *d_out_coords_tangents = nullptr;
    double *d_out_params_tangents = nullptr;
    if(h_in_coords_tangents == nullptr) {
        gpuErrchk(cudaMalloc(&d_out_coords, N*D*sizeof(unsigned long long)));
        gpuErrchk(cudaMemset(d_out_coords, 0, N*D*sizeof(unsigned long long)));
    } else {

        gpuErrchk(cudaMalloc(&d_in_coords_tangents, N*D*sizeof(double)));
        gpuErrchk(cudaMemcpy(d_in_coords_tangents, h_in_coords_tangents, N*D*sizeof(double), cudaMemcpyHostToDevice));

        gpuErrchk(cudaMalloc(&d_out_coords_tangents, N*D*sizeof(double)));
        gpuErrchk(cudaMalloc(&d_out_params_tangents, P*sizeof(double)));

        gpuErrchk(cudaMemset(d_out_coords_tangents, 0, N*D*sizeof(double)));
        gpuErrchk(cudaMemset(d_out_params_tangents, 0, P*sizeof(double)));
    }

    this->execute_device(
        N,
        P,
        d_in_coords, 
        d_in_coords_tangents,
        d_in_params,
        d_out_coords,
        d_out_coords_tangents,
        d_out_params_tangents,
        static_cast<cudaStream_t>(0)
    );

    if(h_in_coords_tangents == nullptr) {
        gpuErrchk(cudaMemcpy(h_out_coords, d_out_coords, N*D*sizeof(*h_out_coords), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaFree(d_out_coords));
    } else {
        gpuErrchk(cudaMemcpy(h_out_coords_tangents, d_out_coords_tangents, N*D*sizeof(double), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_out_params_tangents, d_out_params_tangents, P*sizeof(double), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaFree(d_out_coords_tangents));
        gpuErrchk(cudaFree(d_out_params_tangents));
    }

    gpuErrchk(cudaFree(d_in_coords));
    gpuErrchk(cudaFree(d_in_params));

};

template class Gradient<4>; 
template class Gradient<3>;

}

