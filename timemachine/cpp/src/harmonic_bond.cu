#include "gpu_utils.cuh"
#include "harmonic_bond.hpp"
#include "k_harmonic_bond.cuh"
#include "kernel_utils.cuh"
#include "math_utils.cuh"
#include <cub/cub.cuh>
#include <vector>

namespace timemachine {

template <typename RealType>
HarmonicBond<RealType>::HarmonicBond(const std::vector<int> &bond_idxs)
    : B_(bond_idxs.size() / 2), sum_storage_bytes_(0) {

    if (bond_idxs.size() % 2 != 0) {
        throw std::runtime_error("bond_idxs.size() must be exactly 2*k!");
    }

    for (int b = 0; b < B_; b++) {
        auto src = bond_idxs[b * 2 + 0];
        auto dst = bond_idxs[b * 2 + 1];
        if (src == dst) {
            throw std::runtime_error("src == dst");
        }
    }

    gpuErrchk(cudaGraphCreate(&graph_, 0));

    cudaSafeMalloc(&d_bond_idxs_, B_ * 2 * sizeof(*d_bond_idxs_));
    gpuErrchk(cudaMemcpy(d_bond_idxs_, &bond_idxs[0], B_ * 2 * sizeof(*d_bond_idxs_), cudaMemcpyHostToDevice));
    cudaSafeMalloc(&d_u_buffer_, B_ * sizeof(*d_u_buffer_));

    gpuErrchk(cub::DeviceReduce::Sum(nullptr, sum_storage_bytes_, d_u_buffer_, d_u_buffer_, B_));

    gpuErrchk(cudaMalloc(&d_sum_temp_storage_, sum_storage_bytes_));

    const int tpb = DEFAULT_THREADS_PER_BLOCK;
    const int blocks = ceil_divide(B_, tpb);

    // Set up the harmonic bond kernel, values need to be adjusted at runtime.
    bonded_params_.func = reinterpret_cast<void *>(k_harmonic_bond<RealType>);
    bonded_params_.gridDim = dim3(blocks, 1, 1);
    bonded_params_.blockDim = dim3(tpb, 1, 1);
    bonded_params_.extra = nullptr;
    bonded_params_.sharedMemBytes = 0;
};

template <typename RealType> HarmonicBond<RealType>::~HarmonicBond() {
    gpuErrchk(cudaFree(d_bond_idxs_));
    gpuErrchk(cudaFree(d_u_buffer_));
    gpuErrchk(cudaFree(d_sum_temp_storage_));
    gpuErrchk(cudaGraphExecDestroy(graph_exec_));
    gpuErrchk(cudaGraphDestroy(graph_));
};

template <typename RealType>
void HarmonicBond<RealType>::execute_device(
    const int N,
    const int P,
    const double *d_x,
    const double *d_p,
    const double *d_box,
    unsigned long long *d_du_dx,
    unsigned long long *d_du_dp,
    __int128 *d_u,
    cudaStream_t stream) {

    if (P != 2 * B_) {
        throw std::runtime_error(
            "HarmonicBond::execute_device(): expected P == 2*B, got P=" + std::to_string(P) +
            ", 2*B=" + std::to_string(2 * B_));
    }

    if (B_ > 0) {
        __int128 *d_u_buffer = d_u == nullptr ? nullptr : d_u_buffer_;

        void *kernelArgs[7] = {
            const_cast<void *>(reinterpret_cast<const void *>(&B_)),
            reinterpret_cast<void *>(&d_x),
            reinterpret_cast<void *>(&d_p),
            reinterpret_cast<void *>(&d_bond_idxs_),
            reinterpret_cast<void *>(&d_du_dx),
            reinterpret_cast<void *>(&d_du_dp),
            reinterpret_cast<void *>(&d_u_buffer)};
        bonded_params_.kernelParams = kernelArgs;
        if (graph_exec_ == NULL) {
            gpuErrchk(cudaGraphAddKernelNode(&bonded_node_, graph_, NULL, 0, &bonded_params_));
            gpuErrchk(cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0));
        } else {
            gpuErrchk(cudaGraphExecKernelNodeSetParams(graph_exec_, bonded_node_, &bonded_params_));
        }

        // gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaGraphLaunch(graph_exec_, stream));

        if (d_u) {
            gpuErrchk(cub::DeviceReduce::Sum(d_sum_temp_storage_, sum_storage_bytes_, d_u_buffer_, d_u, B_, stream));
        }
    }
};

template class HarmonicBond<double>;
template class HarmonicBond<float>;

} // namespace timemachine
