#include "nonbonded.hpp"

namespace timemachine {

template <typename RealType, bool Interpolated>
Nonbonded<RealType, Interpolated>::Nonbonded(
    const std::vector<int> &exclusion_idxs,     // [M, 2]
    const std::vector<double> &scales,          // [M, 2]
    const std::vector<int> &lambda_plane_idxs,  // N
    const std::vector<int> &lambda_offset_idxs, // N
    const double beta,
    const double cutoff,
    const std::string &kernel_src)
    : dense_(lambda_plane_idxs, lambda_offset_idxs, beta, cutoff, kernel_src),
      exclusions_(exclusion_idxs, scales, lambda_plane_idxs, lambda_offset_idxs, beta, cutoff, kernel_src) {}

template <typename RealType, bool Interpolated> void Nonbonded<RealType, Interpolated>::set_nblist_padding(double val) {
    dense_.set_nblist_padding(val);
}

template <typename RealType, bool Interpolated> void Nonbonded<RealType, Interpolated>::disable_hilbert_sort() {
    dense_.disable_hilbert_sort();
}

template <typename RealType, bool Interpolated>
void Nonbonded<RealType, Interpolated>::execute_device(
    const int N,
    const int P,
    const double *d_x,
    const double *d_p,
    const double *d_box,
    const double lambda,
    unsigned long long *d_du_dx,
    double *d_du_dp,
    unsigned long long *d_du_dl,
    unsigned long long *d_u,
    cudaStream_t stream) {
    dense_.execute_device(N, P, d_x, d_p, d_box, lambda, d_du_dx, d_du_dp, d_du_dl, d_u, stream);
    exclusions_.execute_device(N, P, d_x, d_p, d_box, lambda, d_du_dx, d_du_dp, d_du_dl, d_u, stream);
};

template class Nonbonded<double, true>;
template class Nonbonded<float, true>;
template class Nonbonded<double, false>;
template class Nonbonded<float, false>;

} // namespace timemachine
