#include "gpu_utils.cuh"
#include "harmonic_angle.hpp"
#include "k_harmonic_angle.cuh"
#include "kernel_utils.cuh"
#include "math_utils.cuh"
#include <vector>

namespace timemachine {

template <typename RealType>
HarmonicAngle<RealType>::HarmonicAngle(const std::vector<int> &angle_idxs // [A, 3]
                                       )
    : A_(angle_idxs.size() / 3) {

    if (angle_idxs.size() % 3 != 0) {
        throw std::runtime_error("angle_idxs.size() must be exactly 3*A");
    }

    for (int a = 0; a < A_; a++) {
        auto i = angle_idxs[a * 3 + 0];
        auto j = angle_idxs[a * 3 + 1];
        auto k = angle_idxs[a * 3 + 2];
        if (i == j || j == k || i == k) {
            throw std::runtime_error("angle triplets must be unique");
        }
    }

    cudaSafeMalloc(&d_angle_idxs_, A_ * 3 * sizeof(*d_angle_idxs_));
    gpuErrchk(cudaMemcpy(d_angle_idxs_, &angle_idxs[0], A_ * 3 * sizeof(*d_angle_idxs_), cudaMemcpyHostToDevice));
};

template <typename RealType> HarmonicAngle<RealType>::~HarmonicAngle() { gpuErrchk(cudaFree(d_angle_idxs_)); };

template <typename RealType>
void HarmonicAngle<RealType>::execute_device(
    const int N,
    const int P,
    const double *d_x,
    const double *d_p,
    const double *d_box,
    unsigned long long *d_du_dx,
    unsigned long long *d_du_dp,
    unsigned long long *d_u,
    cudaStream_t stream) {

    const int tpb = DEFAULT_THREADS_PER_BLOCK;
    const int blocks = ceil_divide(A_, tpb);

    if (A_ > 0) {

        if (P != A_ * 2) {
            throw std::runtime_error(
                "HarmonicAngle::execute_device(): expected P == 2*A_, got P=" + std::to_string(P) +
                ", 2*A_=" + std::to_string(2 * A_));
        }
        k_harmonic_angle<RealType, 3><<<blocks, tpb, 0, stream>>>(A_, d_x, d_p, d_angle_idxs_, d_du_dx, d_du_dp, d_u);
        gpuErrchk(cudaPeekAtLastError());
    }
}

template class HarmonicAngle<double>;
template class HarmonicAngle<float>;

} // namespace timemachine
