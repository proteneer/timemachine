#include "gpu_utils.cuh"
#include "k_lambda_transformer.cuh"
#include "k_nonbonded_pair_list.cuh"
#include "math_utils.cuh"
#include "nonbonded_pair_list.hpp"
#include <stdexcept>
#include <vector>

namespace timemachine {

template <typename RealType, bool Negated>
NonbondedPairList<RealType, Negated>::NonbondedPairList(
    const std::vector<int> &pair_idxs,          // [M, 2]
    const std::vector<double> &scales,          // [M, 2]
    const std::vector<int> &lambda_plane_idxs,  // [N]
    const std::vector<int> &lambda_offset_idxs, // [N]
    const double beta,
    const double cutoff)
    : N_(lambda_offset_idxs.size()), M_(pair_idxs.size() / 2), beta_(beta), cutoff_(cutoff) {

    if (pair_idxs.size() % 2 != 0) {
        throw std::runtime_error("pair_idxs.size() must be even, but got " + std::to_string(pair_idxs.size()));
    }

    for (int i = 0; i < M_; i++) {
        auto src = pair_idxs[i * 2 + 0];
        auto dst = pair_idxs[i * 2 + 1];
        if (src == dst) {
            throw std::runtime_error(
                "illegal pair with src == dst: " + std::to_string(src) + ", " + std::to_string(dst));
        }
    }

    if (scales.size() / 2 != M_) {
        throw std::runtime_error(
            "expected same number of pairs and scale tuples, but got " + std::to_string(M_) +
            " != " + std::to_string(scales.size() / 2));
    }

    gpuErrchk(cudaMalloc(&d_pair_idxs_, M_ * 2 * sizeof(*d_pair_idxs_)));
    gpuErrchk(cudaMemcpy(d_pair_idxs_, &pair_idxs[0], M_ * 2 * sizeof(*d_pair_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_lambda_plane_idxs_, N_ * sizeof(*d_lambda_plane_idxs_)));
    gpuErrchk(cudaMemcpy(
        d_lambda_plane_idxs_, &lambda_plane_idxs[0], N_ * sizeof(*d_lambda_plane_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_lambda_offset_idxs_, N_ * sizeof(*d_lambda_offset_idxs_)));
    gpuErrchk(cudaMemcpy(
        d_lambda_offset_idxs_, &lambda_offset_idxs[0], N_ * sizeof(*d_lambda_offset_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_w_, N_ * sizeof(*d_w_)));

    gpuErrchk(cudaMalloc(&d_scales_, M_ * 2 * sizeof(*d_scales_)));
    gpuErrchk(cudaMemcpy(d_scales_, &scales[0], M_ * 2 * sizeof(*d_scales_), cudaMemcpyHostToDevice));
};

template <typename RealType, bool Negated> NonbondedPairList<RealType, Negated>::~NonbondedPairList() {
    gpuErrchk(cudaFree(d_pair_idxs_));
    gpuErrchk(cudaFree(d_scales_));
    gpuErrchk(cudaFree(d_lambda_plane_idxs_));
    gpuErrchk(cudaFree(d_lambda_offset_idxs_));
    gpuErrchk(cudaFree(d_w_));
};

template <typename RealType, bool Negated>
void NonbondedPairList<RealType, Negated>::execute_device(
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

    const int tpb = 32;

    int num_blocks = ceil_divide(N, tpb);
    dim3 dimGrid(num_blocks, 3, 1);

    k_compute_w_coords<<<dimGrid, tpb, 0, stream>>>(
        N, lambda, cutoff_, d_lambda_plane_idxs_, d_lambda_offset_idxs_, d_w_);
    gpuErrchk(cudaPeekAtLastError());

    int num_blocks_pairs = ceil_divide(M_, tpb);

    k_nonbonded_pair_list<RealType, Negated><<<num_blocks_pairs, tpb, 0, stream>>>(
        M_, d_x, d_p, d_box, d_w_, d_pair_idxs_, d_scales_, beta_, cutoff_, d_du_dx, d_du_dp, d_u);

    gpuErrchk(cudaPeekAtLastError());
}

// TODO: this implementation is duplicated from NonbondedAllPairs
template <typename RealType, bool Negated>
void NonbondedPairList<RealType, Negated>::du_dp_fixed_to_float(
    const int N, const int P, const unsigned long long *du_dp, double *du_dp_float) {

    for (int i = 0; i < N; i++) {
        const int idx_charge = i * 3 + 0;
        const int idx_sig = i * 3 + 1;
        const int idx_eps = i * 3 + 2;
        du_dp_float[idx_charge] = FIXED_TO_FLOAT_DU_DP<double, FIXED_EXPONENT_DU_DCHARGE>(du_dp[idx_charge]);
        du_dp_float[idx_sig] = FIXED_TO_FLOAT_DU_DP<double, FIXED_EXPONENT_DU_DSIG>(du_dp[idx_sig]);
        du_dp_float[idx_eps] = FIXED_TO_FLOAT_DU_DP<double, FIXED_EXPONENT_DU_DEPS>(du_dp[idx_eps]);
    }
}

template class NonbondedPairList<double, true>;
template class NonbondedPairList<float, true>;

template class NonbondedPairList<double, false>;
template class NonbondedPairList<float, false>;

} // namespace timemachine
