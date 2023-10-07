#include "k_hilbert.cuh"

namespace timemachine {

// k_coords_to_kv_gather converts the coords and boxes to floats for performance
// and does not impact the precision of the kernels.
void __global__ k_coords_to_kv_gather(
    const int N,
    const unsigned int *__restrict__ atom_idxs,
    const double *__restrict__ coords,
    const double *__restrict__ box,
    const unsigned int *__restrict__ bin_to_idx,
    unsigned int *__restrict__ keys,
    unsigned int *__restrict__ vals) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) {
        return;
    }

    const int atom_idx = atom_idxs[idx];

    // these coords have to be centered
    float bx = box[0 * 3 + 0];
    float by = box[1 * 3 + 1];
    float bz = box[2 * 3 + 2];

    float binWidth = max(max(bx, by), bz) / (HILBERT_GRID_DIM - 1.0);

    float x = coords[atom_idx * 3 + 0];
    float y = coords[atom_idx * 3 + 1];
    float z = coords[atom_idx * 3 + 2];

    // floor is used in place of nearbyint here to ensure all particles are imaged into the home box. This differs
    // from distances calculations where the nearest possible image is calculated rather than imaging into
    // the home box.
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

} // namespace timemachine
