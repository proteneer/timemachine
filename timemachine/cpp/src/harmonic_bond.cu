#include "gpu_utils.cuh"
#include "harmonic_bond.hpp"
#include "k_harmonic_bond.cuh"
#include "kernel_utils.cuh"
#include "math_utils.cuh"
#include <vector>

namespace timemachine {

template <typename RealType>
HarmonicBond<RealType>::HarmonicBond(const std::vector<int> &bond_idxs) : B_(bond_idxs.size() / 2) {

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

    gpuErrchk(cudaMalloc(&d_bond_idxs_, B_ * 2 * sizeof(*d_bond_idxs_)));
    gpuErrchk(cudaMemcpy(d_bond_idxs_, &bond_idxs[0], B_ * 2 * sizeof(*d_bond_idxs_), cudaMemcpyHostToDevice));
};

template <typename RealType> HarmonicBond<RealType>::~HarmonicBond() { gpuErrchk(cudaFree(d_bond_idxs_)); };

template <typename RealType>
void HarmonicBond<RealType>::execute_device(
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

    if (P != 2 * B_) {
        throw std::runtime_error(
            "HarmonicBond::execute_device(): expected P == 2*B, got P=" + std::to_string(P) +
            ", 2*B=" + std::to_string(2 * B_));
    }

    if (B_ > 0) {
        const int tpb = warp_size;
        const int blocks = ceil_divide(B_, tpb);

        k_harmonic_bond<RealType>
            <<<blocks, tpb, 0, stream>>>(B_, d_x, d_p, lambda, d_bond_idxs_, d_du_dx, d_du_dp, d_du_dl, d_u);
        gpuErrchk(cudaPeekAtLastError());
    }
};

template class HarmonicBond<double>;
template class HarmonicBond<float>;

} // namespace timemachine
