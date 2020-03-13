
#define WARPSIZE 32

void __global__ k_find_block_bounds(
    const int N,
    const int D,
    const int T,
    const double *coords,
    double *block_bounds_ctr,
    double *block_bounds_ext) {

    const int tile_idx = blockDim.x*blockIdx.x + threadIdx.x;

    if(tile_idx >= T) {
        return;
    }

    for(int d=0; d < D; d++) {
        double ci_min =  9999999;
        double ci_max = -9999999;
        for(int i=0; i < WARPSIZE; i++) {
            int atom_i_idx = tile_idx*WARPSIZE + i;
            if(atom_i_idx < N) {
                double ci = coords[atom_i_idx*D + d];
                ci_min = ci < ci_min ? ci : ci_min;
                ci_max = ci > ci_max ? ci : ci_max;                
            }
        }
     
        block_bounds_ctr[tile_idx*D+d] = (ci_max + ci_min)/2.0;
        block_bounds_ext[tile_idx*D+d] = ci_max - ci_min;
    }

}
