#pragma once

namespace timemachine {

// Divide [0,1]^3 box into HILBERT_GRID_DIM^3 voxels for Hilbert sort
static const int HILBERT_GRID_DIM = 128;

// Encode grid index along each dimension using HILBERT_N_BITS
static const int HILBERT_N_BITS = 8;
static const int HILBERT_MAX_GRID_DIM = 1 << HILBERT_N_BITS;
static_assert(HILBERT_N_BITS == 8);
static_assert(HILBERT_GRID_DIM <= HILBERT_MAX_GRID_DIM);

// generate kv values from coordinates to be radix sorted allowing the selection of a subset of coordinates
void __global__ k_coords_to_kv_gather(
    const int N,
    const unsigned int *__restrict__ atom_idxs,
    const double *__restrict__ coords,
    const double *__restrict__ box,
    const unsigned int *__restrict__ bin_to_idx,
    unsigned int *__restrict__ keys,
    unsigned int *__restrict__ vals);

} // namespace timemachine
