#include <chrono>
#include <iostream>
#include <vector>
#include <complex>
#include "harmonic_bond.hpp"
#include "gpu_utils.cuh"
#include "k_bonded.cuh"

namespace timemachine {

template <typename RealType>
HarmonicBond<RealType>::HarmonicBond(
    const std::vector<int> &bond_idxs, // [N]
    const std::vector<double> &params
) : B_(bond_idxs.size()/2) {

    if(bond_idxs.size() % 2 != 0) {
        throw std::runtime_error("bond_idxs.size() must be exactly 2*k!");
    }

    if(bond_idxs.size() != params.size()) {
        throw std::runtime_error("bond_idxs.size() must match params.size()!");
    }

    for(int b=0; b < B_; b++) {
        auto src = bond_idxs[b*2+0];
        auto dst = bond_idxs[b*2+1];
        if(src == dst) {
            throw std::runtime_error("src == dst");
        }
    }

    gpuErrchk(cudaMalloc(&d_bond_idxs_, B_*2*sizeof(*d_bond_idxs_)));
    gpuErrchk(cudaMemcpy(d_bond_idxs_, &bond_idxs[0], B_*2*sizeof(*d_bond_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_params_, B_*2*sizeof(*d_params_)));
    gpuErrchk(cudaMemcpy(d_params_, &params[0], B_*2*sizeof(*d_params_), cudaMemcpyHostToDevice));

    // it's critical that we initialize these, this also implies that this class now holds state
    // (which is okay)
    gpuErrchk(cudaMalloc(&d_du_dp_primals_, B_*2*sizeof(*d_du_dp_primals_)));
    gpuErrchk(cudaMemset(d_du_dp_primals_, 0, B_*2*sizeof(*d_du_dp_primals_)));

    gpuErrchk(cudaMalloc(&d_du_dp_tangents_, B_*2*sizeof(*d_du_dp_tangents_)));
    gpuErrchk(cudaMemset(d_du_dp_tangents_, 0, B_*2*sizeof(*d_du_dp_tangents_)));

    // gpuErrchk(cudaMemcpy(d_du_dp_primals_, &params[0], B_*2*sizeof(*d_params_), cudaMemcpyHostToDevice));

};

template <typename RealType>
HarmonicBond<RealType>::~HarmonicBond() {
    gpuErrchk(cudaFree(d_bond_idxs_));

    gpuErrchk(cudaFree(d_params_));
    gpuErrchk(cudaFree(d_du_dp_primals_));
    gpuErrchk(cudaFree(d_du_dp_tangents_));
};

template <typename RealType>
void HarmonicBond<RealType>::get_du_dp_primals(double *buf) {
    gpuErrchk(cudaMemcpy(buf, d_du_dp_primals_, B_*2*sizeof(*d_params_), cudaMemcpyDeviceToHost));
}

template <typename RealType>
void HarmonicBond<RealType>::get_du_dp_tangents(double *buf) {
    gpuErrchk(cudaMemcpy(buf, d_du_dp_tangents_, B_*2*sizeof(*d_params_), cudaMemcpyDeviceToHost));
}

template <typename RealType>
void HarmonicBond<RealType>::execute_lambda_inference_device(
    const int N,
    // const int P,
    const double *d_coords_primals,
    const double lambda_primal,
    unsigned long long *d_out_coords_primals, // du/dx
    double *d_out_lambda_primals, // du/dl
    double *d_out_energy_primal, // U
    cudaStream_t stream) {
    int tpb = 32;
    int blocks = (B_+tpb-1)/tpb;
    k_harmonic_bond_inference<RealType><<<blocks, tpb, 0, stream>>>(
        B_,
        d_coords_primals,
        d_params_,
        d_bond_idxs_,
        d_out_coords_primals,
        d_out_energy_primal
    );
    gpuErrchk(cudaPeekAtLastError());

    // auto finish = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = finish - start;
    // std::cout << "HarmonicBond Elapsed time: " << elapsed.count() << " s\n";

};

template <typename RealType>
void HarmonicBond<RealType>::execute_lambda_jvp_device(
    const int N,
    // const int P,
    const double *d_coords_primals,
    const double *d_coords_tangents,
    const double lambda_primal, // unused
    const double lambda_tangent, // unused
    double *d_out_coords_primals,
    double *d_out_coords_tangents,
    // double *d_out_params_primals,
    // double *d_out_params_tangents,
    cudaStream_t stream) {

    int tpb = 32;
    int blocks = (B_+tpb-1)/tpb;

    k_harmonic_bond_jvp<RealType><<<blocks, tpb, 0, stream>>>(
        B_,
        d_coords_primals,
        d_coords_tangents,
        d_params_,
        d_bond_idxs_,
        d_out_coords_primals,
        d_out_coords_tangents,
        d_du_dp_primals_,
        d_du_dp_tangents_
    );

    // cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    // auto finish = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = finish - start;
    // std::cout << "HarmonicBond Elapsed time: " << elapsed.count() << " s\n";

}

template class HarmonicBond<double>;
template class HarmonicBond<float>;

} // namespace timemachine