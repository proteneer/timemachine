#include <chrono>
#include <iostream>
#include <vector>
#include <complex>
#include "centroid_restraint.hpp"
#include "gpu_utils.cuh"
#include "k_centroid_restraint.cuh"

namespace timemachine {

template <typename RealType>
CentroidRestraint<RealType>::CentroidRestraint(
    const int N,
    const std::vector<int> &group_a_idxs,
    const std::vector<int> &group_b_idxs,
    const double kb,
    const double b0
) : N_(N),
    N_A_(group_a_idxs.size()),
    N_B_(group_b_idxs.size()),
    kb_(kb),
    b0_(b0) {

    for(int i=0; i < group_a_idxs.size(); i++) {
        if(group_a_idxs[i] >= N_ || group_a_idxs[i] < 0) {
            throw std::runtime_error("Invalid group_a_idx!");
        }
    }

    for(int i=0; i < group_b_idxs.size(); i++) {
        if(group_b_idxs[i] >= N_ || group_b_idxs[i] < 0) {
            throw std::runtime_error("Invalid group_a_idx!");
        }
    }

    gpuErrchk(cudaMalloc(&d_group_a_idxs_, N_A_*sizeof(*d_group_a_idxs_)));
    gpuErrchk(cudaMemcpy(d_group_a_idxs_, &group_a_idxs[0], N_A_*sizeof(*d_group_a_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_group_b_idxs_, N_B_*sizeof(*d_group_b_idxs_)));
    gpuErrchk(cudaMemcpy(d_group_b_idxs_, &group_b_idxs[0], N_B_*sizeof(*d_group_b_idxs_), cudaMemcpyHostToDevice));

};

template <typename RealType>
CentroidRestraint<RealType>::~CentroidRestraint() {
    gpuErrchk(cudaFree(d_group_a_idxs_));
    gpuErrchk(cudaFree(d_group_b_idxs_));
};


template <typename RealType>
void CentroidRestraint<RealType>::execute_device(
        const int N,
        const int P,
        const double *d_x,
        const double *d_p,
        const double *d_box,
        const double lambda,
        unsigned long long *d_du_dx,
        double *d_du_dp,
        unsigned long long *d_du_dl,
        unsigned long long *d_u,
        cudaStream_t stream) {

    int tpb = 32;

    k_centroid_restraint<RealType><<<1, tpb, 0, stream>>>(
        N_,
        d_x,
        d_group_a_idxs_,
        d_group_b_idxs_,
        N_A_,
        N_B_,
        // d_masses_,
        kb_,
        b0_,
        d_du_dx,
        d_u
    );
    gpuErrchk(cudaPeekAtLastError());

    // auto finish = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = finish - start;
    // std::cout << "CentroidRestraint Elapsed time: " << elapsed.count() << " s\n";

};

template class CentroidRestraint<double>;
template class CentroidRestraint<float>;

} // namespace timemachine