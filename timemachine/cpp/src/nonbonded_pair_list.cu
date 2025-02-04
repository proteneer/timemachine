#include "energy_accumulation.hpp"
#include "gpu_utils.cuh"
#include "k_nonbonded_pair_list.cuh"
#include "kernels/kernel_utils.cuh"
#include "math_utils.cuh"
#include "nonbonded_pair_list.hpp"
#include <stdexcept>
#include <vector>

namespace timemachine {

template <typename RealType, bool Negated>
NonbondedPairList<RealType, Negated>::NonbondedPairList(
    const std::vector<int> &pair_idxs, // [M, 2]
    const std::vector<double> &scales, // [M, 2]
    const double beta,
    const double cutoff)
    : M_(pair_idxs.size() / 2), beta_(beta), cutoff_(cutoff), sum_storage_bytes_(0) {

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

    cudaSafeMalloc(&d_u_buffer_, M_ * sizeof(*d_u_buffer_));

    cudaSafeMalloc(&d_pair_idxs_, M_ * 2 * sizeof(*d_pair_idxs_));
    gpuErrchk(cudaMemcpy(d_pair_idxs_, &pair_idxs[0], M_ * 2 * sizeof(*d_pair_idxs_), cudaMemcpyHostToDevice));

    cudaSafeMalloc(&d_scales_, M_ * 2 * sizeof(*d_scales_));
    gpuErrchk(cudaMemcpy(d_scales_, &scales[0], M_ * 2 * sizeof(*d_scales_), cudaMemcpyHostToDevice));
};

template <typename RealType, bool Negated> NonbondedPairList<RealType, Negated>::~NonbondedPairList() {
    gpuErrchk(cudaFree(d_pair_idxs_));
    gpuErrchk(cudaFree(d_scales_));
    gpuErrchk(cudaFree(d_u_buffer_));
};

template <typename RealType, bool Negated>
void NonbondedPairList<RealType, Negated>::execute_device(
    const int N,
    const int P,
    const double *d_x,
    const double *d_p,
    const double *d_box,
    unsigned long long *d_du_dx,
    unsigned long long *d_du_dp,
    __int128 *d_u,
    cudaStream_t stream) {

    if (M_ > 0) {
        const int tpb = DEFAULT_THREADS_PER_BLOCK;
        const int num_blocks_pairs = ceil_divide(M_, tpb);

        k_nonbonded_pair_list<RealType, Negated><<<num_blocks_pairs, tpb, 0, stream>>>(
            M_,
            d_x,
            d_p,
            d_box,
            d_pair_idxs_,
            d_scales_,
            beta_,
            cutoff_,
            d_du_dx,
            d_du_dp,
            d_u == nullptr ? nullptr : d_u_buffer_);

        gpuErrchk(cudaPeekAtLastError());

        if (d_u) {
            accumulate_energy(M_, d_u_buffer_, d_u, stream);
        }
    }
}

// TODO: this implementation is duplicated from NonbondedAllPairs
template <typename RealType, bool Negated>
void NonbondedPairList<RealType, Negated>::du_dp_fixed_to_float(
    const int N, const int P, const unsigned long long *du_dp, double *du_dp_float) {

    for (int i = 0; i < N; i++) {
        const int idx = i * PARAMS_PER_ATOM;
        const int idx_charge = idx + PARAM_OFFSET_CHARGE;
        const int idx_sig = idx + PARAM_OFFSET_SIG;
        const int idx_eps = idx + PARAM_OFFSET_EPS;
        const int idx_w = idx + PARAM_OFFSET_W;

        du_dp_float[idx_charge] = FIXED_TO_FLOAT_DU_DP<double, FIXED_EXPONENT_DU_DCHARGE>(du_dp[idx_charge]);
        du_dp_float[idx_sig] = FIXED_TO_FLOAT_DU_DP<double, FIXED_EXPONENT_DU_DSIG>(du_dp[idx_sig]);
        du_dp_float[idx_eps] = FIXED_TO_FLOAT_DU_DP<double, FIXED_EXPONENT_DU_DEPS>(du_dp[idx_eps]);
        du_dp_float[idx_w] = FIXED_TO_FLOAT_DU_DP<double, FIXED_EXPONENT_DU_DW>(du_dp[idx_w]);
    }
}

template class NonbondedPairList<double, true>;
template class NonbondedPairList<float, true>;

template class NonbondedPairList<double, false>;
template class NonbondedPairList<float, false>;

} // namespace timemachine
