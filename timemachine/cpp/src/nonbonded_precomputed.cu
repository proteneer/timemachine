#include "gpu_utils.cuh"
#include "k_nonbonded_precomputed.cuh"
#include "kernel_utils.cuh"
#include "math_utils.cuh"
#include "nonbonded_precomputed.hpp"
#include <vector>

namespace timemachine {

template <typename RealType>
NonbondedPairListPrecomputed<RealType>::NonbondedPairListPrecomputed(
    const std::vector<int> &idxs, const std::vector<double> &w_offsets, const double beta, const double cutoff)
    : B_(idxs.size() / 2), beta_(beta), cutoff_(cutoff) {

    if (idxs.size() % 2 != 0) {
        throw std::runtime_error("idxs.size() must be exactly 2*k!");
    }

    for (int b = 0; b < B_; b++) {
        auto src = idxs[b * 2 + 0];
        auto dst = idxs[b * 2 + 1];
        if (src == dst) {
            throw std::runtime_error("src == dst");
        }
    }

    gpuErrchk(cudaMalloc(&d_idxs_, B_ * 2 * sizeof(*d_idxs_)));
    gpuErrchk(cudaMemcpy(d_idxs_, &idxs[0], B_ * 2 * sizeof(*d_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_w_offsets_, B_ * sizeof(*d_w_offsets_)));
    gpuErrchk(cudaMemcpy(d_w_offsets_, &w_offsets[0], B_ * sizeof(*d_w_offsets_), cudaMemcpyHostToDevice));
};

template <typename RealType> NonbondedPairListPrecomputed<RealType>::~NonbondedPairListPrecomputed() {
    gpuErrchk(cudaFree(d_idxs_));
    gpuErrchk(cudaFree(d_w_offsets_));
};

template <typename RealType>
void NonbondedPairListPrecomputed<RealType>::execute_device(
    const int N,
    const int P,
    const double *d_x,
    const double *d_p,
    const double *d_box,
    const double lambda,
    unsigned long long *d_du_dx,
    unsigned long long *d_du_dp,
    unsigned long long *d_du_dl,
    unsigned long long *d_u,
    cudaStream_t stream) {

    if (P != 3 * B_) {
        throw std::runtime_error(
            "NonbondedPairListPrecomputed::execute_device(): expected P == 2*B, got P=" + std::to_string(P) +
            ", 2*B=" + std::to_string(2 * B_));
    }

    if (B_ > 0) {
        const int tpb = warp_size;
        const int blocks = ceil_divide(B_, tpb);

        k_nonbonded_precomputed<RealType><<<blocks, tpb, 0, stream>>>(
            B_, d_x, d_p, d_box, d_w_offsets_, d_idxs_, beta_, cutoff_, d_du_dx, d_du_dp, d_u);
    }
};

template <typename RealType>
void NonbondedPairListPrecomputed<RealType>::du_dp_fixed_to_float(
    const int N, const int P, const unsigned long long *du_dp, double *du_dp_float) {

    // In the interpolated case we have derivatives for the initial and final parameters
    const int num_tuples = B_;

    for (int i = 0; i < num_tuples; i++) {
        const int idx_charge = i * 3 + 0;
        const int idx_sig = i * 3 + 1;
        const int idx_eps = i * 3 + 2;

        du_dp_float[idx_charge] = FIXED_TO_FLOAT_DU_DP<double, FIXED_EXPONENT_DU_DCHARGE>(du_dp[idx_charge]);
        du_dp_float[idx_sig] = FIXED_TO_FLOAT_DU_DP<double, FIXED_EXPONENT_DU_DSIG>(du_dp[idx_sig]);
        du_dp_float[idx_eps] = FIXED_TO_FLOAT_DU_DP<double, FIXED_EXPONENT_DU_DEPS>(du_dp[idx_eps]);
    }
};

template class NonbondedPairListPrecomputed<double>;
template class NonbondedPairListPrecomputed<float>;

} // namespace timemachine
