#include "energy_accumulation.hpp"
#include "gpu_utils.cuh"
#include "k_nonbonded_precomputed.cuh"
#include "kernel_utils.cuh"
#include "kernels/k_nonbonded_common.cuh"
#include "math_utils.cuh"
#include "nonbonded_precomputed.hpp"
#include <vector>

namespace timemachine {

template <typename RealType>
NonbondedPairListPrecomputed<RealType>::NonbondedPairListPrecomputed(
    const std::vector<int> &idxs, const double beta, const double cutoff)
    : B_(idxs.size() / 2), beta_(beta), cutoff_(cutoff) {

    if (idxs.size() % 2 != 0) {
        throw std::runtime_error("idxs.size() must be exactly 2*B!");
    }

    for (int b = 0; b < B_; b++) {
        auto src = idxs[b * 2 + 0];
        auto dst = idxs[b * 2 + 1];
        if (src == dst) {
            throw std::runtime_error(
                "illegal pair with src == dst: " + std::to_string(src) + ", " + std::to_string(dst));
        }
    }

    cudaSafeMalloc(&d_idxs_, B_ * 2 * sizeof(*d_idxs_));
    gpuErrchk(cudaMemcpy(d_idxs_, &idxs[0], B_ * 2 * sizeof(*d_idxs_), cudaMemcpyHostToDevice));

    cudaSafeMalloc(&d_u_buffer_, B_ * sizeof(*d_u_buffer_));
};

template <typename RealType> NonbondedPairListPrecomputed<RealType>::~NonbondedPairListPrecomputed() {
    gpuErrchk(cudaFree(d_idxs_));
    gpuErrchk(cudaFree(d_u_buffer_));
};

template <typename RealType>
void NonbondedPairListPrecomputed<RealType>::execute_device(
    const int N,
    const int P,
    const double *d_x,
    const double *d_p,
    const double *d_box,
    unsigned long long *d_du_dx,
    unsigned long long *d_du_dp,
    __int128 *d_u,
    cudaStream_t stream) {

    if (P != PARAMS_PER_PAIR * B_) {
        throw std::runtime_error(
            "NonbondedPairListPrecomputed::execute_device(): expected P == " + std::to_string(PARAMS_PER_PAIR) +
            "*B, got P=" + std::to_string(P) + ", " + std::to_string(PARAMS_PER_PAIR) +
            "*B=" + std::to_string(PARAMS_PER_PAIR * B_));
    }

    if (B_ > 0) {
        const int tpb = DEFAULT_THREADS_PER_BLOCK;
        const int blocks = ceil_divide(B_, tpb);

        k_nonbonded_precomputed<RealType><<<blocks, tpb, 0, stream>>>(
            B_,
            d_x,
            d_p,
            d_box,
            d_idxs_,
            static_cast<RealType>(beta_),
            static_cast<RealType>(cutoff_ * cutoff_),
            d_du_dx,
            d_du_dp,
            d_u == nullptr ? nullptr : d_u_buffer_);
        gpuErrchk(cudaPeekAtLastError());

        if (d_u) {
            accumulate_energy(B_, d_u_buffer_, d_u, stream);
        }
    }
};

template <typename RealType>
void NonbondedPairListPrecomputed<RealType>::du_dp_fixed_to_float(
    const int N, const int P, const unsigned long long *du_dp, double *du_dp_float) {

    for (int i = 0; i < B_; i++) {
        const int idx = i * PARAMS_PER_PAIR;
        const int idx_charge = idx + PARAM_OFFSET_CHARGE;
        const int idx_sig = idx + PARAM_OFFSET_SIG;
        const int idx_eps = idx + PARAM_OFFSET_EPS;
        const int idx_w = idx + PARAM_OFFSET_W;

        du_dp_float[idx_charge] = FIXED_TO_FLOAT_DU_DP<double, FIXED_EXPONENT_DU_DCHARGE>(du_dp[idx_charge]);
        du_dp_float[idx_sig] = FIXED_TO_FLOAT_DU_DP<double, FIXED_EXPONENT_DU_DSIG>(du_dp[idx_sig]);
        du_dp_float[idx_eps] = FIXED_TO_FLOAT_DU_DP<double, FIXED_EXPONENT_DU_DEPS>(du_dp[idx_eps]);
        du_dp_float[idx_w] = FIXED_TO_FLOAT_DU_DP<double, FIXED_EXPONENT_DU_DW>(du_dp[idx_w]);
    }
};

template class NonbondedPairListPrecomputed<double>;
template class NonbondedPairListPrecomputed<float>;

} // namespace timemachine
