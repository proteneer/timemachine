#pragma once

#include <array>
#include <vector>

namespace timemachine {

template <typename RealType>
std::array<std::vector<int>, 2> get_inner_and_outer_mols(
    const std::vector<int> &center_atoms,
    const std::vector<double> &coords,
    const std::vector<double> &box,
    const std::vector<std::vector<int>> &group_idxs,
    const RealType radius);

} // namespace timemachine
