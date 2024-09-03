
#include "k_nonbonded_common.cuh"

namespace timemachine {

void __global__ k_update_neighborlist_state(
    const int N,
    const int *__restrict__ rebuild_flag, // [1]
    const double *__restrict__ coords,    // [N, 3]
    const double *__restrict__ box,       // [3, 3]
    double *__restrict__ nblist_coords,   // [N, 3]
    double *__restrict__ nblist_box       // [3, 3]
) {
    const int D = 3;
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (*rebuild_flag == 0) {
        return;
    }

    while (block_idx < N * D) {
        if (block_idx < D * D) {
            nblist_box[block_idx] = box[block_idx];
        }
        nblist_coords[block_idx] = coords[block_idx];

        block_idx += gridDim.x * blockDim.x;
    }
}

} // namespace timemachine
