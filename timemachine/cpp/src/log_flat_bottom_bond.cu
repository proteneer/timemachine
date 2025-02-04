#include "energy_accumulation.hpp"
#include "gpu_utils.cuh"
#include "k_log_flat_bottom_bond.cuh"
#include "kernel_utils.cuh"
#include "log_flat_bottom_bond.hpp"
#include "math_utils.cuh"
#include <cub/cub.cuh>
#include <vector>

namespace timemachine {

template <typename RealType>
LogFlatBottomBond<RealType>::LogFlatBottomBond(const std::vector<int> &bond_idxs, double beta)
    : B_(bond_idxs.size() / 2), beta_(beta), sum_storage_bytes_(0) {

    if (beta <= 0) {
        throw std::runtime_error("beta must be positive");
    }
    // (TODO): deboggle
    // validate bond_idxs: even length, all idxs non-negative, and no self-edges
    if (bond_idxs.size() % 2 != 0) {
        throw std::runtime_error("bond_idxs.size() must be exactly 2*k!");
    }

    for (int b = 0; b < B_; b++) {
        auto src = bond_idxs[b * 2 + 0];
        auto dst = bond_idxs[b * 2 + 1];
        if (src == dst) {
            throw std::runtime_error("src == dst");
        }

        if ((src < 0) or (dst < 0)) {
            throw std::runtime_error("idxs must be non-negative");
        }
    }

    // copy idxs to device
    cudaSafeMalloc(&d_bond_idxs_, B_ * 2 * sizeof(*d_bond_idxs_));
    gpuErrchk(cudaMemcpy(d_bond_idxs_, &bond_idxs[0], B_ * 2 * sizeof(*d_bond_idxs_), cudaMemcpyHostToDevice));
    cudaSafeMalloc(&d_u_buffer_, B_ * sizeof(*d_u_buffer_));

    gpuErrchk(cub::DeviceReduce::Sum(nullptr, sum_storage_bytes_, d_u_buffer_, d_u_buffer_, B_));

    gpuErrchk(cudaMalloc(&d_sum_temp_storage_, sum_storage_bytes_));
};

template <typename RealType> LogFlatBottomBond<RealType>::~LogFlatBottomBond() {
    gpuErrchk(cudaFree(d_bond_idxs_));
    gpuErrchk(cudaFree(d_u_buffer_));
    gpuErrchk(cudaFree(d_sum_temp_storage_));
};

template <typename RealType>
void LogFlatBottomBond<RealType>::execute_device(
    const int N,
    const int P,
    const double *d_x,
    const double *d_p,
    const double *d_box,
    unsigned long long *d_du_dx,
    unsigned long long *d_du_dp,
    __int128 *d_u,
    cudaStream_t stream) {

    const int num_params_per_bond = 3;
    int expected_P = num_params_per_bond * B_;

    if (P != expected_P) {
        throw std::runtime_error(
            "LogFlatBottomBond::execute_device(): expected P == " + std::to_string(expected_P) +
            ", got P=" + std::to_string(P));
    }

    if (B_ > 0) {
        const int tpb = DEFAULT_THREADS_PER_BLOCK;
        const int blocks = ceil_divide(B_, tpb);

        k_log_flat_bottom_bond<RealType><<<blocks, tpb, 0, stream>>>(
            B_, d_x, d_box, d_p, d_bond_idxs_, beta_, d_du_dx, d_du_dp, d_u == nullptr ? nullptr : d_u_buffer_);
        gpuErrchk(cudaPeekAtLastError());

        if (d_u) {
            gpuErrchk(cub::DeviceReduce::Sum(d_sum_temp_storage_, sum_storage_bytes_, d_u_buffer_, d_u, B_, stream));
        }
    }
};

template <typename RealType>
void LogFlatBottomBond<RealType>::set_bonds_device(const int num_bonds, const int *d_bonds, const cudaStream_t stream) {
    gpuErrchk(cudaMemcpyAsync(
        d_bond_idxs_, d_bonds, num_bonds * 2 * sizeof(*d_bond_idxs_), cudaMemcpyDeviceToDevice, stream));
    B_ = num_bonds;
}

template class LogFlatBottomBond<double>;
template class LogFlatBottomBond<float>;

} // namespace timemachine
