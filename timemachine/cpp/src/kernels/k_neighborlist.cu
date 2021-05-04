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

    // Algorithm taken from https://github.com/openmm/openmm/blob/master/platforms/cuda/src/kernels/findInteractingBlocks.cu#L7
    // Computes smaller bounding boxes than simpler form by accounting for periodic box conditions

    // each thread processes one tile
    const int index = blockIdx.x*blockDim.x+threadIdx.x;
    if(index >= T) {
        return;
    }

    const double bx = box[0*3+0];
    const double by = box[1*3+1];
    const double bz = box[2*3+2];

    const double inv_bx = 1/bx;
    const double inv_by = 1/by;
    const double inv_bz = 1/bz;
    const int base = index*TILESIZE;

    double pos_x = coords[base*3+0];
    double pos_y = coords[base*3+1];
    double pos_z = coords[base*3+2];

    double minPos_x = pos_x;
    double minPos_y = pos_y;
    double minPos_z = pos_z;

    double maxPos_x = pos_x;
    double maxPos_y = pos_y;
    double maxPos_z = pos_z;
    int last = min(base+TILESIZE, N);
    for (int i = base+1; i < last; i++) {
        pos_x = coords[i*3+0];
        pos_y = coords[i*3+1];
        pos_z = coords[i*3+2];
        // Build up center over time, and recenter before computing
        // min and max, to reduce overall size of box thanks to accounting
        // for periodic boundary conditions
        double center_x = 0.5f*(maxPos_x+minPos_x);
        double center_y = 0.5f*(maxPos_y+minPos_y);
        double center_z = 0.5f*(maxPos_z+minPos_z);
        pos_x -= bx*nearbyint((pos_x-center_x)*inv_bx);
        pos_y -= by*nearbyint((pos_y-center_y)*inv_by);
        pos_z -= bz*nearbyint((pos_z-center_z)*inv_bz);
        minPos_x = min(minPos_x,pos_x);
        minPos_y = min(minPos_y,pos_y);
        minPos_z = min(minPos_z,pos_z);

        maxPos_x = max(maxPos_x,pos_x);
        maxPos_y = max(maxPos_y,pos_y);
        maxPos_z = max(maxPos_z,pos_z);
    }
    block_bounds_ext[index*3+0] = 0.5f*(maxPos_x-minPos_x);
    block_bounds_ext[index*3+1] = 0.5f*(maxPos_y-minPos_y);
    block_bounds_ext[index*3+2] = 0.5f*(maxPos_z-minPos_z);

    block_bounds_ctr[index*3+0] = 0.5f*(maxPos_x+minPos_x);
    block_bounds_ctr[index*3+1] = 0.5f*(maxPos_y+minPos_y);
    block_bounds_ctr[index*3+2] = 0.5f*(maxPos_z+minPos_z);
}
