#include <cstdio>

#define WARPSIZE 32


void __global__ k_find_block_bounds(
    const int N,
    const int D,
    const int T,
    const double *coords,
    const double *box,
    // const int *perm,
    double *block_bounds_ctr,
    double *block_bounds_ext) {

    // each thread processes one tile
    const int tile_idx = blockDim.x*blockIdx.x + threadIdx.x;

    if(tile_idx >= T) {
        return;
    }

    for(int d=0; d < D; d++) {
        double width = box[d*3+d];
        double ci_min =  9999999;
        double ci_max = -9999999;
        for(int i=0; i < WARPSIZE; i++) {
            int tid = tile_idx*WARPSIZE + i;
            if(tid < N) {
                // int atom_idx = perm[tid];
                int atom_idx = tid;
                double ci = coords[atom_idx*D + d];
                ci -= width*floor(ci/width); // move to home box
                ci_min = ci < ci_min ? ci : ci_min;
                ci_max = ci > ci_max ? ci : ci_max;
            }
        }
     
        // printf("dim %d tile_idx %d ctr %f ext %f\n", d, tile_idx, (ci_max + ci_min)/2.0, ci_max - ci_min);
        block_bounds_ctr[tile_idx*D+d] = (ci_max + ci_min)/2.0;
        block_bounds_ext[tile_idx*D+d] = (ci_max - ci_min)/2.0;
    }

}
