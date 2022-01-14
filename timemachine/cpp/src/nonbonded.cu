#include "nonbonded.hpp"
#include <vector>

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
      exclusions_(
          exclusion_idxs, negate_scales_(scales), lambda_plane_idxs, lambda_offset_idxs, beta, cutoff, kernel_src) {}

template <typename RealType, bool Interpolated>
std::vector<double> Nonbonded<RealType, Interpolated>::negate_scales_(const std::vector<double> &scales) {
    std::vector<double> negated(scales.size());
    for (int i = 0; i < negated.size(); i++) {
        negated[i] = -scales[i];
    }
    return negated;
}

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
    unsigned long long *d_du_dp,
    unsigned long long *d_du_dl,
    unsigned long long *d_u,
    cudaStream_t stream) {
    dense_.execute_device(N, P, d_x, d_p, d_box, lambda, d_du_dx, d_du_dp, d_du_dl, d_u, stream);
    exclusions_.execute_device(N, P, d_x, d_p, d_box, lambda, d_du_dx, d_du_dp, d_du_dl, d_u, stream);
};

template <typename RealType, bool Interpolated>
void Nonbonded<RealType, Interpolated>::du_dp_fixed_to_float(
    const int N, const int P, const unsigned long long *du_dp, double *du_dp_float) {
    dense_.du_dp_fixed_to_float(N, P, du_dp, du_dp_float);
}

template class Nonbonded<double, true>;
template class Nonbonded<float, true>;
template class Nonbonded<double, false>;
template class Nonbonded<float, false>;

} // namespace timemachine
