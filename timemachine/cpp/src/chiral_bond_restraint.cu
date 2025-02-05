#include "chiral_bond_restraint.hpp"
#include "gpu_utils.cuh"
#include "k_chiral_restraint.cuh"
#include "kernel_utils.cuh"
#include "math_utils.cuh"
#include <cub/cub.cuh>
#include <vector>

namespace timemachine {

template <typename RealType>
ChiralBondRestraint<RealType>::ChiralBondRestraint(const std::vector<int> &idxs, const std::vector<int> &signs)
    : R_(idxs.size() / 4), sum_storage_bytes_(0) {

    if (idxs.size() % 4 != 0) {
        throw std::runtime_error("idxs.size() must be exactly 4*R!");
    }

    if (R_ != signs.size()) {
        throw std::runtime_error("signs.size() must be exactly R!");
    }

    for (auto s : signs) {
        if (s != -1 && s != 1) {
            throw std::runtime_error("signs must be comprised exclusively of 1 or -1");
        }
    }

    cudaSafeMalloc(&d_idxs_, R_ * 4 * sizeof(*d_idxs_));
    gpuErrchk(cudaMemcpy(d_idxs_, &idxs[0], R_ * 4 * sizeof(*d_idxs_), cudaMemcpyHostToDevice));

    cudaSafeMalloc(&d_signs_, R_ * sizeof(*d_signs_));
    gpuErrchk(cudaMemcpy(d_signs_, &signs[0], R_ * sizeof(*d_signs_), cudaMemcpyHostToDevice));

    cudaSafeMalloc(&d_u_buffer_, R_ * sizeof(*d_u_buffer_));

    gpuErrchk(cub::DeviceReduce::Sum(nullptr, sum_storage_bytes_, d_u_buffer_, d_u_buffer_, R_));

    gpuErrchk(cudaMalloc(&d_sum_temp_storage_, sum_storage_bytes_));
};

template <typename RealType> ChiralBondRestraint<RealType>::~ChiralBondRestraint() {
    gpuErrchk(cudaFree(d_idxs_));
    gpuErrchk(cudaFree(d_signs_));
    gpuErrchk(cudaFree(d_u_buffer_));
    gpuErrchk(cudaFree(d_sum_temp_storage_));
};

template <typename RealType>
void ChiralBondRestraint<RealType>::execute_device(
    const int N,
    const int P,
    const double *d_x,
    const double *d_p,
    const double *d_box,
    unsigned long long *d_du_dx,
    unsigned long long *d_du_dp,
    __int128 *d_u,
    cudaStream_t stream) {

    if (P != R_) {
        throw std::runtime_error(
            "ChiralBondRestraint::execute_device(): expected P == R, got P=" + std::to_string(P) +
            ", R=" + std::to_string(R_));
    }

    if (R_ > 0) {
        const int tpb = DEFAULT_THREADS_PER_BLOCK;
        const int blocks = ceil_divide(R_, tpb);

        k_chiral_bond_restraint<RealType><<<blocks, tpb, 0, stream>>>(
            R_, d_x, d_p, d_idxs_, d_signs_, d_du_dx, d_du_dp, d_u == nullptr ? nullptr : d_u_buffer_);
        gpuErrchk(cudaPeekAtLastError());

        if (d_u) {
            gpuErrchk(cub::DeviceReduce::Sum(d_sum_temp_storage_, sum_storage_bytes_, d_u_buffer_, d_u, R_, stream));
        }
    }
};

template class ChiralBondRestraint<double>;
template class ChiralBondRestraint<float>;

} // namespace timemachine
