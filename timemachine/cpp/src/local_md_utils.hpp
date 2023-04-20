#pragma once
#include <memory>
#include <typeinfo>

#include "bound_potential.hpp"
#include "potential.hpp"

namespace timemachine {

bool is_nonbonded_all_pairs_potential(std::shared_ptr<Potential> pot);

void set_nonbonded_potential_idxs(
    std::shared_ptr<Potential> pot, const int num_idxs, const unsigned int *d_idxs, const cudaStream_t stream);

void set_nonbonded_ixn_potential_idxs(
    std::shared_ptr<Potential> pot,
    const int num_col_idxs,
    const int num_row_idxs,
    unsigned int *d_col_idxs,
    unsigned int *d_row_idxs,
    const cudaStream_t stream);

std::shared_ptr<BoundPotential>
construct_ixn_group_potential(const int N, std::shared_ptr<Potential> pot, const int P, const double *d_params);

// Recursively populate nb_pots potentials with the NonbondedAllPairs
void get_nonbonded_all_pair_potentials(
    std::vector<std::shared_ptr<BoundPotential>> input, std::vector<std::shared_ptr<BoundPotential>> &flattened);

void verify_local_md_parameters(double radius, double k);

} // namespace timemachine
