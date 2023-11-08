#pragma once

#include <vector>

namespace timemachine {

template <typename RealType>
std::vector<RealType> compute_atom_by_atom_energies(
    const int N,
    const std::vector<int> &target_atoms,
    const std::vector<double> &coords,
    const std::vector<double> &params,
    std::vector<double> &box,
    const RealType nb_beta,
    const RealType cutoff);

} // namespace timemachine
