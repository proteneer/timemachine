#include "k_hilbert.cuh"

namespace timemachine {

// k_coords_to_kv_gather converts the coords and boxes to floats for performance
// and does not impact the precision of the kernels.
// Note that this kernel requires the use of double precision as imaging into the home box
// as expected, with float precision can be outside of the home box for coordinates with large magnitudes
void __global__ k_coords_to_kv_gather(
    const int N,
    const unsigned int *__restrict__ atom_idxs,
    const double *__restrict__ coords,
    const double *__restrict__ box,
    const unsigned int *__restrict__ bin_to_idx,
    unsigned int *__restrict__ keys,
    unsigned int *__restrict__ vals) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const double bx = box[0 * 3 + 0];
    const double by = box[1 * 3 + 1];
    const double bz = box[2 * 3 + 2];

    const double inv_bx = 1 / bx;
    const double inv_by = 1 / by;
    const double inv_bz = 1 / bz;

    const double inv_bin_width = min(min(inv_bx, inv_by), inv_bz) * (HILBERT_GRID_DIM - 1.0);

    while (idx < N) {
        int atom_idx = atom_idxs[idx];

        double x = coords[atom_idx * 3 + 0];
        double y = coords[atom_idx * 3 + 1];
        double z = coords[atom_idx * 3 + 2];

        // floor is used in place of nearbyint here to ensure all particles are imaged into the home box. This differs
        // from distances calculations where the nearest possible image is calculated rather than imaging into
        // the home box.
        x -= bx * floor(x * inv_bx);
        y -= by * floor(y * inv_by);
        z -= bz * floor(z * inv_bz);

        unsigned int bin_x = static_cast<unsigned int>(x * inv_bin_width);
        unsigned int bin_y = static_cast<unsigned int>(y * inv_bin_width);
        unsigned int bin_z = static_cast<unsigned int>(z * inv_bin_width);

        keys[idx] = bin_to_idx[bin_x * HILBERT_GRID_DIM * HILBERT_GRID_DIM + bin_y * HILBERT_GRID_DIM + bin_z];
        // uncomment below if you want to preserve the atom ordering
        // keys[idx] = atom_idx;
        vals[idx] = atom_idx;

        idx += gridDim.x * blockDim.x;
    }
}

} // namespace timemachine
