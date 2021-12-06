#include "gpu_utils.cuh"
#include "k_nonbonded_pairs.cuh"
#include "nonbonded_pairs.hpp"
#include <stdexcept>
#include <vector>

namespace timemachine {

template <typename RealType>
NonbondedPairs<RealType>::NonbondedPairs(
    const std::vector<int> &pair_idxs, // [M, 2]
    const std::vector<double> &scales, // [M, 2]
    const double beta,
    const double cutoff)
    : M_(pair_idxs.size() / 2), beta_(beta), cutoff_(cutoff) {

    if (pair_idxs.size() % 2 != 0) {
        throw std::runtime_error("pair_idxs.size() must be exactly 2*M");
    }

    for (int i = 0; i < M_; i++) {
        auto src = pair_idxs[i * 2 + 0];
        auto dst = pair_idxs[i * 2 + 1];
        if (src == dst) {
            throw std::runtime_error("illegal pair with src == dst");
        }
    }

    gpuErrchk(cudaMalloc(&d_pair_idxs_, M_ * 2 * sizeof(*d_pair_idxs_)));
    gpuErrchk(cudaMemcpy(d_pair_idxs_, &pair_idxs[0], M_ * 2 * sizeof(*d_pair_idxs_), cudaMemcpyHostToDevice));

    if (scales.size() / 2 != M_) {
        throw std::runtime_error("bad scales size!");
    }

    gpuErrchk(cudaMalloc(&d_scales_, M_ * 2 * sizeof(*d_scales_)));
    gpuErrchk(cudaMemcpy(d_scales_, &scales[0], M_ * 2 * sizeof(*d_scales_), cudaMemcpyHostToDevice));
};

template <typename RealType> NonbondedPairs<RealType>::~NonbondedPairs() { gpuErrchk(cudaFree(d_pair_idxs_)); };

template <typename RealType>
void NonbondedPairs<RealType>::execute_device(
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

    const int tpb = 32;

    dim3 dimGridExclusions((M_ + tpb - 1) / tpb, 1, 1);

    k_nonbonded_pairs<RealType><<<dimGridExclusions, tpb, 0, stream>>>(
        M_,
        d_x,
        d_p,
        d_box,
        d_dp_dl_,
        d_w_,
        d_dw_dl_,
        d_pair_idxs_,
        d_scales_,
        beta_,
        cutoff_,
        d_du_dx,
        d_du_dp_buffer_,
        d_du_dl,
        d_u);

    gpuErrchk(cudaPeekAtLastError());
}

template class NonbondedPairs<double>;
template class NonbondedPairs<float>;

} // namespace timemachine
