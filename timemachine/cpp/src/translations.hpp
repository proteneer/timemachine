#pragma once
#include <vector>

namespace timemachine {

template <typename RealType>
std::vector<RealType> get_translations_inside_and_outside_sphere_host(
    const int n_translations,
    const std::vector<double> &box,
    const std::vector<RealType> &center,
    const RealType radius,
    const int seed);

template <typename RealType>
std::vector<RealType> get_translations_outside_sphere_host(
    const int n_translations,

    const std::vector<RealType> &center,
    const RealType radius,
    const int seed);

} // namespace timemachine
