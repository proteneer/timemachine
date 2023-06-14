#pragma once

#include <vector>

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
    unsigned long long *__restrict__ u_buffer);

void verify_atom_idxs(int N, const std::vector<int> &atom_idxs, const bool allow_empty = false);
