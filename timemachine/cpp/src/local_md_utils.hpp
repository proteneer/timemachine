#pragma once
#include <memory>
#include <typeinfo>

#include "bound_potential.hpp"
#include "potential.hpp"

namespace timemachine {

void set_nonbonded_potential_idxs(
    std::shared_ptr<Potential> pot, const int num_idxs, const unsigned int *d_idxs, const cudaStream_t stream);

// Copies the atom indices of the nonbonded all pairs potential and returns the number of indices copied.
int copy_nonbonded_potential_idxs(std::shared_ptr<Potential> pot, const int max_idxs, unsigned int *d_output_idxs);

void set_nonbonded_ixn_potential_idxs(
    std::shared_ptr<Potential> pot,
    const int num_col_idxs,
    const int num_row_idxs,
    unsigned int *d_col_idxs,
    unsigned int *d_row_idxs,
    const cudaStream_t stream);

std::shared_ptr<BoundPotential> construct_ixn_group_potential(
    const int N, std::shared_ptr<Potential> pot, const int P, const double *d_params, const double nblist_padding);

void verify_local_md_parameters(double radius, double k);

} // namespace timemachine
