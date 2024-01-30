#pragma once
#include <vector>

namespace timemachine {

template <typename RealType>
std::vector<RealType> translations_inside_and_outside_sphere_host(
    const int n_translations,
    const std::vector<double> &box,
    const std::vector<RealType> &center,
    const RealType radius,
    const int seed);

} // namespace timemachine
