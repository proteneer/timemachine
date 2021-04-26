#include <cstdio>

#define TILESIZE 32


void __global__ k_find_block_bounds(
    const int N, // Number of atoms
    const int D, // Box dimensions, typically 3
    const int T, // Number of tiles
    const double *coords, // [N*3]
    const double *box, // [D*3]
    double * __restrict__ block_bounds_ctr,
    double * __restrict__ block_bounds_ext) {

    // each thread processes one tile
    const int tile_idx = blockDim.x*blockIdx.x + threadIdx.x;

    if(tile_idx >= T) {
        return;
    }

    for(int d=0; d < D; d++) {
        const double width = box[d*3+d];
        double ci_min =  9999999;
        double ci_max = -9999999;
        for(int i=0; i < TILESIZE; i++) {
            const int atom_idx = tile_idx*TILESIZE + i;
            if(atom_idx < N) {
                double ci = coords[atom_idx*D + d];
                ci -= width*floor(ci/width); // move to home box
                ci_min = ci < ci_min ? ci : ci_min;
                ci_max = ci > ci_max ? ci : ci_max;
            }
        }

        block_bounds_ctr[tile_idx*D+d] = (ci_max + ci_min)/2.0;
        block_bounds_ext[tile_idx*D+d] = (ci_max - ci_min)/2.0;
    }
}
