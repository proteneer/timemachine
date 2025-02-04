#include "energy_accumulation.hpp"
#include "gpu_utils.cuh"
#include "k_periodic_torsion.cuh"
#include "kernel_utils.cuh"
#include "math_utils.cuh"
#include "periodic_torsion.hpp"
#include <cub/cub.cuh>
#include <vector>

namespace timemachine {

template <typename RealType>
PeriodicTorsion<RealType>::PeriodicTorsion(const std::vector<int> &torsion_idxs // [A, 4]
                                           )
    : T_(torsion_idxs.size() / 4), sum_storage_bytes_(0) {

    if (torsion_idxs.size() % 4 != 0) {
        throw std::runtime_error("torsion_idxs.size() must be exactly 4*k");
    }

    for (int a = 0; a < T_; a++) {
        auto i = torsion_idxs[a * 4 + 0];
        auto j = torsion_idxs[a * 4 + 1];
        auto k = torsion_idxs[a * 4 + 2];
        auto l = torsion_idxs[a * 4 + 3];
        if (i == j || i == k || i == l || j == k || j == l || k == l) {
            throw std::runtime_error("torsion quads must be unique");
        }
    }

    cudaSafeMalloc(&d_torsion_idxs_, T_ * 4 * sizeof(*d_torsion_idxs_));
    gpuErrchk(cudaMemcpy(d_torsion_idxs_, &torsion_idxs[0], T_ * 4 * sizeof(*d_torsion_idxs_), cudaMemcpyHostToDevice));

    cudaSafeMalloc(&d_u_buffer_, T_ * sizeof(*d_u_buffer_));

    gpuErrchk(cub::DeviceReduce::Sum(nullptr, sum_storage_bytes_, d_u_buffer_, d_u_buffer_, T_));

    gpuErrchk(cudaMalloc(&d_sum_temp_storage_, sum_storage_bytes_));
};

template <typename RealType> PeriodicTorsion<RealType>::~PeriodicTorsion() {
    gpuErrchk(cudaFree(d_torsion_idxs_));
    gpuErrchk(cudaFree(d_u_buffer_));
    gpuErrchk(cudaFree(d_sum_temp_storage_));
};

template <typename RealType>
void PeriodicTorsion<RealType>::execute_device(
    const int N,
    const int P,
    const double *d_x,
    const double *d_p,
    const double *d_box,
    unsigned long long *d_du_dx,
    unsigned long long *d_du_dp,
    __int128 *d_u,
    cudaStream_t stream) {

    const int tpb = DEFAULT_THREADS_PER_BLOCK;
    const int blocks = ceil_divide(T_, tpb);

    const int D = 3;

    if (blocks > 0) {
        if (P != 3 * T_) {
            throw std::runtime_error(
                "PeriodicTorsion::execute_device(): expected P == 3*T_, got P=" + std::to_string(P) +
                ", 3*T_=" + std::to_string(3 * T_));
        }
        k_periodic_torsion<RealType, D><<<blocks, tpb, 0, stream>>>(
            T_, d_x, d_p, d_torsion_idxs_, d_du_dx, d_du_dp, d_u == nullptr ? nullptr : d_u_buffer_);
        gpuErrchk(cudaPeekAtLastError());

        if (d_u) {
            gpuErrchk(cub::DeviceReduce::Sum(d_sum_temp_storage_, sum_storage_bytes_, d_u_buffer_, d_u, T_, stream));
        }
    }
};

template class PeriodicTorsion<double>;
template class PeriodicTorsion<float>;

} // namespace timemachine
