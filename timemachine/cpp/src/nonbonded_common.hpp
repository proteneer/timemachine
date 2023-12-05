#pragma once

#include <memory>
#include <vector>

#include "bound_potential.hpp"
#include "potential.hpp"

namespace timemachine {

// each atom parameterized by a 4-tuple: charge, lj sigma, lj epsilon, 4D coordinate w
enum { PARAM_OFFSET_CHARGE = 0, PARAM_OFFSET_SIG, PARAM_OFFSET_EPS, PARAM_OFFSET_W, PARAMS_PER_ATOM };

typedef void (*k_nonbonded_fn)(
    const int N,
    const int NR,
    const unsigned int *ixn_count,
    const double *__restrict__ coords,
    const double *__restrict__ params, // [N]
    const double *__restrict__ box,
    const double beta,
    const double cutoff,
    const unsigned int *__restrict__ row_idxs,
    const int *__restrict__ ixn_tiles,
    const unsigned int *__restrict__ ixn_atoms,
    unsigned long long *__restrict__ du_dx,
    unsigned long long *__restrict__ du_dp,
    __int128 *__restrict__ u_buffer);

void verify_atom_idxs(int N, const std::vector<int> &atom_idxs, const bool allow_empty = false);

// Recursively populate nb_pots potentials with the NonbondedAllPairs
void get_nonbonded_all_pair_potentials(
    std::vector<std::shared_ptr<BoundPotential>> input, std::vector<std::shared_ptr<BoundPotential>> &flattened);

double get_nonbonded_all_pair_cutoff_with_padding(std::shared_ptr<Potential> pot);

} // namespace timemachine
