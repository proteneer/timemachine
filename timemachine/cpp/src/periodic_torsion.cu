#include <chrono>
#include <iostream>
#include <vector>
#include <complex>
#include "periodic_torsion.hpp"
#include "gpu_utils.cuh"
#include "k_bonded.cuh"

namespace timemachine {

template <typename RealType>
PeriodicTorsion<RealType>::PeriodicTorsion(
    const std::vector<int> &torsion_idxs, // [A, 4]
    const std::vector<double> &params // [A, 3]
) : T_(torsion_idxs.size()/4) {

    if(torsion_idxs.size() % 4 != 0) {
        throw std::runtime_error("torsion_idxs.size() must be exactly 4*k");
    }

    for(int a=0; a < T_; a++) {
        auto i = torsion_idxs[a*4+0];
        auto j = torsion_idxs[a*4+1];
        auto k = torsion_idxs[a*4+2];
        auto l = torsion_idxs[a*4+3];
        if(i == j || i == k || i == l || j == k || j == l || k == l) {
            throw std::runtime_error("torsion quads must be unique");
        }
    }

    gpuErrchk(cudaMalloc(&d_torsion_idxs_, T_*4*sizeof(*d_torsion_idxs_)));
    gpuErrchk(cudaMemcpy(d_torsion_idxs_, &torsion_idxs[0], T_*4*sizeof(*d_torsion_idxs_), cudaMemcpyHostToDevice));

    // gpuErrchk(cudaMalloc(&d_param_idxs_, T_*3*sizeof(*d_param_idxs_)));
    // gpuErrchk(cudaMemcpy(d_param_idxs_, &param_idxs[0], T_*3*sizeof(*d_param_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_params_, T_*3*sizeof(*d_params_)));
    gpuErrchk(cudaMemcpy(d_params_, &params[0], T_*3*sizeof(*d_params_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_du_dp_primals_, T_*3*sizeof(*d_du_dp_primals_)));
    gpuErrchk(cudaMemset(d_du_dp_primals_, 0, T_*3*sizeof(*d_du_dp_primals_)));

    gpuErrchk(cudaMalloc(&d_du_dp_tangents_, T_*3*sizeof(*d_du_dp_tangents_)));
    gpuErrchk(cudaMemset(d_du_dp_tangents_, 0, T_*3*sizeof(*d_du_dp_tangents_)));

};

template <typename RealType>
PeriodicTorsion<RealType>::~PeriodicTorsion() {
    gpuErrchk(cudaFree(d_torsion_idxs_));
    // gpuErrchk(cudaFree(d_param_idxs_));

    gpuErrchk(cudaFree(d_params_));
    gpuErrchk(cudaFree(d_du_dp_primals_));
    gpuErrchk(cudaFree(d_du_dp_tangents_));

};

template <typename RealType>
void PeriodicTorsion<RealType>::get_du_dp_primals(double *buf) {
    gpuErrchk(cudaMemcpy(buf, d_du_dp_primals_, T_*3*sizeof(*d_params_), cudaMemcpyDeviceToHost));
}

template <typename RealType>
void PeriodicTorsion<RealType>::get_du_dp_tangents(double *buf) {
    gpuErrchk(cudaMemcpy(buf, d_du_dp_tangents_, T_*3*sizeof(*d_params_), cudaMemcpyDeviceToHost));
}

template <typename RealType>
void PeriodicTorsion<RealType>::execute_lambda_inference_device(
    const int N,
    // const int P,
    const double *d_coords_primals,
    // const double *d_params_primals,
    const double lambda_primal,
    unsigned long long *d_out_coords_primals, // du/dx
    double *d_out_lambda_primal, // du/dl, unused
    double *d_out_energy_primal, // U
    cudaStream_t stream) {

    int tpb = 32;
    int blocks = (T_+tpb-1)/tpb;

    const int D = 3;

    k_periodic_torsion_inference<RealType, D><<<blocks, tpb, 0, stream>>>(
        T_,
        d_coords_primals,
        // d_params_primals,
        d_params_,
        d_torsion_idxs_,
        // d_param_idxs_,
        d_out_coords_primals,
        d_out_energy_primal
    );

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

};

template <typename RealType>
void PeriodicTorsion<RealType>::execute_lambda_jvp_device(
    const int N,
    // const int P,
    const double *d_coords_primals,
    const double *d_coords_tangents,
    // const double *d_params_primals,
    const double lambda_primal, // unused
    const double lambda_tangent, // unused
    double *d_out_coords_primals,
    double *d_out_coords_tangents,
    // double *d_out_params_primals,
    // double *d_out_params_tangents,
    cudaStream_t stream) {

    int tpb = 32;
    int blocks = (T_+tpb-1)/tpb;
    const int D = 3;
    k_periodic_torsion_jvp<RealType, D><<<blocks, tpb, 0, stream>>>(
        T_,
        d_coords_primals,
        d_coords_tangents,
        d_params_,
        // d_params_primals,
        d_torsion_idxs_,
        // d_param_idxs_,
        d_out_coords_primals,
        d_out_coords_tangents,
        d_du_dp_primals_,
        d_du_dp_tangents_
    );

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

};

template class PeriodicTorsion<double>;
template class PeriodicTorsion<float>;

} // namespace timemachine