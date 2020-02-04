
#include "gradient.hpp"
#include "kernel_utils.cuh"
#include "surreal.cuh"

namespace timemachine {

template<typename RealType, int D>
void Gradient<RealType, D>::execute_host(
    const int N,
    const int P,
    const RealType *h_in_coords,
    const RealType *h_in_coords_tangents,
    const RealType *h_in_params,
    unsigned long long *h_out_coords,
    RealType *h_out_coords_tangents,
    RealType *h_out_params_tangents) {

    RealType *d_in_coords;
    RealType *d_in_params;

    gpuErrchk(cudaMalloc(&d_in_coords, N*D*sizeof(RealType)));
    gpuErrchk(cudaMalloc(&d_in_params, P*sizeof(RealType)));
    gpuErrchk(cudaMemcpy(d_in_coords, h_in_coords, N*D*sizeof(RealType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_in_params, h_in_params, P*sizeof(RealType), cudaMemcpyHostToDevice));

    unsigned long long *d_out_coords;

    // very important that we initialize these
    RealType *d_in_coords_tangents = nullptr;
    RealType *d_out_coords_tangents = nullptr;
    RealType *d_out_params_tangents = nullptr;
    if(h_in_coords_tangents == nullptr) {
        gpuErrchk(cudaMalloc(&d_out_coords, N*D*sizeof(unsigned long long)));
        gpuErrchk(cudaMemset(d_out_coords, 0, N*D*sizeof(unsigned long long)));
    } else {

        gpuErrchk(cudaMalloc(&d_in_coords_tangents, N*D*sizeof(RealType)));
        gpuErrchk(cudaMemcpy(d_in_coords_tangents, h_in_coords_tangents, N*D*sizeof(RealType), cudaMemcpyHostToDevice));

        gpuErrchk(cudaMalloc(&d_out_coords_tangents, N*D*sizeof(RealType)));
        gpuErrchk(cudaMalloc(&d_out_params_tangents, P*sizeof(RealType)));

        gpuErrchk(cudaMemset(d_out_coords_tangents, 0, N*D*sizeof(RealType)));
        gpuErrchk(cudaMemset(d_out_params_tangents, 0, P*sizeof(RealType)));
    }

    this->execute_device(
        N,
        P,
        d_in_coords, 
        d_in_coords_tangents,
        d_in_params,
        d_out_coords,
        d_out_coords_tangents,
        d_out_params_tangents
    );

    if(h_in_coords_tangents == nullptr) {
        gpuErrchk(cudaMemcpy(h_out_coords, d_out_coords, N*D*sizeof(RealType), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaFree(d_out_coords));
    } else {
        gpuErrchk(cudaMemcpy(h_out_coords_tangents, d_out_coords_tangents, N*D*sizeof(RealType), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_out_params_tangents, d_out_params_tangents, P*sizeof(RealType), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaFree(d_out_coords_tangents));
        gpuErrchk(cudaFree(d_out_params_tangents));
    }

    gpuErrchk(cudaFree(d_in_coords));
    gpuErrchk(cudaFree(d_in_params));

};

template class Gradient<double, 4>; 
template class Gradient<double, 3>;

}

