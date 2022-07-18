#include "flat_bottom_bond.hpp"
#include "gpu_utils.cuh"
#include "k_flat_bottom_bond.cuh"
#include "kernel_utils.cuh"
#include "math_utils.cuh"
#include <vector>

namespace timemachine {

template <typename RealType>
FlatBottomBond<RealType>::FlatBottomBond(const std::vector<int> &bond_idxs) : B_(bond_idxs.size() / 2) {

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
    gpuErrchk(cudaMalloc(&d_bond_idxs_, B_ * 2 * sizeof(*d_bond_idxs_)));
    gpuErrchk(cudaMemcpy(d_bond_idxs_, &bond_idxs[0], B_ * 2 * sizeof(*d_bond_idxs_), cudaMemcpyHostToDevice));
};

template <typename RealType> FlatBottomBond<RealType>::~FlatBottomBond() { gpuErrchk(cudaFree(d_bond_idxs_)); };

template <typename RealType>
void FlatBottomBond<RealType>::execute_device(
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

    const int num_params_per_bond = 3;
    int expected_P = num_params_per_bond * B_;

    if (P != expected_P) {
        throw std::runtime_error(
            "FlatBottomBond::execute_device(): expected P == " + std::to_string(expected_P) +
            ", got P=" + std::to_string(P));
    }

    if (B_ > 0) {
        const int tpb = warp_size;
        const int blocks = ceil_divide(B_, tpb);

        k_flat_bottom_bond<RealType>
            <<<blocks, tpb, 0, stream>>>(B_, d_x, d_box, d_p, d_bond_idxs_, d_du_dx, d_du_dp, d_u);
        gpuErrchk(cudaPeekAtLastError());
    }
};

template class FlatBottomBond<double>;
template class FlatBottomBond<float>;

} // namespace timemachine
