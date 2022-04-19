#include "k_nonbonded.cuh"

void __global__ k_coords_to_kv(
    const int N,
    const double *coords,
    const double *box,
    const unsigned int *bin_to_idx,
    unsigned int *keys,
    unsigned int *vals) {

    const int atom_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (atom_idx >= N) {
        return;
    }

    // these coords have to be centered
    double bx = box[0 * 3 + 0];
    double by = box[1 * 3 + 1];
    double bz = box[2 * 3 + 2];

    double binWidth = max(max(bx, by), bz) / (HILBERT_GRID_DIM - 1.0);

    double x = coords[atom_idx * 3 + 0];
    double y = coords[atom_idx * 3 + 1];
    double z = coords[atom_idx * 3 + 2];

    x -= bx * floor(x / bx);
    y -= by * floor(y / by);
    z -= bz * floor(z / bz);

    unsigned int bin_x = x / binWidth;
    unsigned int bin_y = y / binWidth;
    unsigned int bin_z = z / binWidth;

    keys[atom_idx] = bin_to_idx[bin_x * HILBERT_GRID_DIM * HILBERT_GRID_DIM + bin_y * HILBERT_GRID_DIM + bin_z];
    // uncomment below if you want to preserve the atom ordering
    // keys[atom_idx] = atom_idx;
    vals[atom_idx] = atom_idx;
}

// TODO: DRY with k_coords_to_kv
void __global__ k_coords_to_kv_gather(
    const int N,
    const unsigned int *atom_idxs,
    const double *coords,
    const double *box,
    const unsigned int *bin_to_idx,
    unsigned int *keys,
    unsigned int *vals) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) {
        return;
    }

    const int atom_idx = atom_idxs[idx];

    // these coords have to be centered
    double bx = box[0 * 3 + 0];
    double by = box[1 * 3 + 1];
    double bz = box[2 * 3 + 2];

    double binWidth = max(max(bx, by), bz) / (HILBERT_GRID_DIM - 1.0);

    double x = coords[atom_idx * 3 + 0];
    double y = coords[atom_idx * 3 + 1];
    double z = coords[atom_idx * 3 + 2];

    x -= bx * floor(x / bx);
    y -= by * floor(y / by);
    z -= bz * floor(z / bz);

    unsigned int bin_x = x / binWidth;
    unsigned int bin_y = y / binWidth;
    unsigned int bin_z = z / binWidth;

    keys[idx] = bin_to_idx[bin_x * HILBERT_GRID_DIM * HILBERT_GRID_DIM + bin_y * HILBERT_GRID_DIM + bin_z];
    // uncomment below if you want to preserve the atom ordering
    // keys[idx] = atom_idx;
    vals[idx] = atom_idx;
}
