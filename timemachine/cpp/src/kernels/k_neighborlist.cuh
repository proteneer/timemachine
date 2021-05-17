#pragma once

#define FULL_MASK 0xffffffff
#define TILESIZE 32
#define WARPSIZE 32

template<typename RealType>
void __global__ k_find_block_bounds(
    const int N, // Number of atoms
    const int D, // Box dimensions, typically 3
    const int T, // Number of tiles
    const double * __restrict__ coords, // [N*3]
    const double * __restrict__ box, // [D*3]
    double *block_bounds_ctr, // [T*3]
    double *block_bounds_ext // [T*3]
) {

    // Algorithm taken from https://github.com/openmm/openmm/blob/master/platforms/cuda/src/kernels/findInteractingBlocks.cu#L7
    // Computes smaller bounding boxes than simpler form by accounting for periodic box conditions

    // each thread processes one tile
    const int index = blockIdx.x*blockDim.x+threadIdx.x;
    if(index >= T) {
        return;
    }

    const RealType bx = box[0*3+0];
    const RealType by = box[1*3+1];
    const RealType bz = box[2*3+2];

    const RealType inv_bx = 1/bx;
    const RealType inv_by = 1/by;
    const RealType inv_bz = 1/bz;
    const int base = index*TILESIZE;

    RealType pos_x = coords[base*3+0];
    RealType pos_y = coords[base*3+1];
    RealType pos_z = coords[base*3+2];

    RealType minPos_x = pos_x;
    RealType minPos_y = pos_y;
    RealType minPos_z = pos_z;

    RealType maxPos_x = pos_x;
    RealType maxPos_y = pos_y;
    RealType maxPos_z = pos_z;

    const int last = min(base+TILESIZE, N);
    for (int i = base+1; i < last; i++) {
        pos_x = coords[i*3+0];
        pos_y = coords[i*3+1];
        pos_z = coords[i*3+2];
        // Build up center over time, and recenter before computing
        // min and max, to reduce overall size of box thanks to accounting
        // for periodic boundary conditions
        RealType center_x = static_cast<RealType>(0.5)*(maxPos_x+minPos_x);
        RealType center_y = static_cast<RealType>(0.5)*(maxPos_y+minPos_y);
        RealType center_z = static_cast<RealType>(0.5)*(maxPos_z+minPos_z);
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

    block_bounds_ctr[index*3+0]  = static_cast<RealType>(0.5)*(maxPos_x+minPos_x);
    block_bounds_ctr[index*3+1]  = static_cast<RealType>(0.5)*(maxPos_y+minPos_y);
    block_bounds_ctr[index*3+2]  = static_cast<RealType>(0.5)*(maxPos_z+minPos_z);

    block_bounds_ext[index*3+0] = static_cast<RealType>(0.5)*(maxPos_x-minPos_x);
    block_bounds_ext[index*3+1] = static_cast<RealType>(0.5)*(maxPos_y-minPos_y);
    block_bounds_ext[index*3+2] = static_cast<RealType>(0.5)*(maxPos_z-minPos_z);
}

void __global__ k_compact_trim_atoms(
    const int N,
    const int Y,
    unsigned int* __restrict__ trim_atoms,
    unsigned int* __restrict__ interactionCount,
    int* __restrict__ interactingTiles,
    unsigned int* __restrict__ interactingAtoms) {

    __shared__ int ixn_j_buffer[64]; // we can probably get away with using only 32 if we do some fancier remainder tricks, but this isn't a huge save
    ixn_j_buffer[threadIdx.x] = N;
    ixn_j_buffer[WARPSIZE+threadIdx.x] = N;

    const int indexInWarp = threadIdx.x%WARPSIZE;
    const int warpMask = (1<<indexInWarp)-1;
    const int row_block_idx = blockIdx.x;

    __shared__ volatile int sync_start[1];
    int neighborsInBuffer = 0;

    for(int trim_block_idx=0; trim_block_idx < Y; trim_block_idx++) {

        int atom_j_idx = trim_atoms[row_block_idx*Y*WARPSIZE + trim_block_idx*WARPSIZE + threadIdx.x];
        bool interacts = atom_j_idx < N;

        int includeAtomFlags = __ballot_sync(FULL_MASK, interacts);

        if (interacts) {
            // only interacting atoms partake in this
            int index = neighborsInBuffer+__popc(includeAtomFlags & warpMask); // where to store this in shared memory
            ixn_j_buffer[index] = atom_j_idx;
        }
        neighborsInBuffer += __popc(includeAtomFlags);

        if(neighborsInBuffer > WARPSIZE) {
            int tilesToStore = 1;
            if(indexInWarp == 0) {
                sync_start[0] = atomicAdd(interactionCount, tilesToStore);
            }
            __syncwarp();
            interactingTiles[sync_start[0]] = row_block_idx; // IS THIS CORRECT? CONTESTED
            interactingAtoms[sync_start[0]*WARPSIZE + threadIdx.x] = ixn_j_buffer[threadIdx.x];

            ixn_j_buffer[threadIdx.x] = ixn_j_buffer[WARPSIZE+threadIdx.x];
            ixn_j_buffer[WARPSIZE+threadIdx.x] = N; // reset old values
            neighborsInBuffer -= WARPSIZE;
        }
    }

    if(neighborsInBuffer > 0) {
        int tilesToStore = 1;
        if(indexInWarp == 0) {
            sync_start[0] = atomicAdd(interactionCount, tilesToStore);
        }
        __syncwarp();
        interactingTiles[sync_start[0]] = row_block_idx;
        interactingAtoms[sync_start[0]*WARPSIZE + threadIdx.x] = ixn_j_buffer[threadIdx.x];
    }

}


/*

This is a simplified algorithm of find_interacting_blocks.cu from OpenMM.
This is launched with a threadblock size of 32, (i.e. one warp).

Each block proceeds as follows:

1. Loads its own row block (32 atoms).
2. Compare the row block against 32 other column blocks via bounding box tests.
3. Determine which blocks potentially interact using warp-level programming.
4. Loop over each interacting block j, and see which row block atoms may interact with j's bbox.
5. For atoms that interact, do a fine-grained comparison of each row block again against each col block atom.
6. Store the column atoms that interact into shared memory
7. Store the leftover trim into its own buffer.

*/

template<typename RealType>
void __global__ k_find_blocks_with_ixns(
    const int N,
    const double * __restrict__ bb_ctr, // [N * 3]
    const double * __restrict__ bb_ext, // [N * 3]
    const double* __restrict__ coords, //TBD make float32 version
    const double* __restrict__ box,
    unsigned int* __restrict__ interactionCount, // number of tiles that have interactions
    int* __restrict__ interactingTiles, // the row block idx of the tile that is interacting
    unsigned int* __restrict__ interactingAtoms, // the col block of the atoms that are interacting
    unsigned int* __restrict__ trim_atoms, // the left-over trims that will later be compacted
    const double cutoff) {

    const int indexInWarp = threadIdx.x%WARPSIZE;
    const int warpMask = (1<<indexInWarp)-1;

    __shared__ int ixn_j_buffer[64]; // we can probably get away with using only 32 if we do some fancier remainder tricks, but this isn't a huge save

    // initialize
    ixn_j_buffer[threadIdx.x] = N;
    ixn_j_buffer[WARPSIZE+threadIdx.x] = N;

    __shared__ volatile int sync_start[1];
    
    const int row_block_idx = blockIdx.x;

    // Retrieve the center coords of row's box and outer limits of row box.
    RealType row_bb_ctr_x = bb_ctr[row_block_idx*3+0];
    RealType row_bb_ctr_y = bb_ctr[row_block_idx*3+1];
    RealType row_bb_ctr_z = bb_ctr[row_block_idx*3+2];

    RealType row_bb_ext_x = bb_ext[row_block_idx*3+0];
    RealType row_bb_ext_y = bb_ext[row_block_idx*3+1];
    RealType row_bb_ext_z = bb_ext[row_block_idx*3+2];

    int neighborsInBuffer = 0;

    const unsigned int atom_i_idx = blockIdx.x*blockDim.x + threadIdx.x;

    RealType pos_i_x = atom_i_idx < N ? coords[atom_i_idx*3 + 0] : 0;
    RealType pos_i_y = atom_i_idx < N ? coords[atom_i_idx*3 + 1] : 0;
    RealType pos_i_z = atom_i_idx < N ? coords[atom_i_idx*3 + 2] : 0;

    const int NUM_BLOCKS = (N+TILESIZE-1)/TILESIZE;

    RealType bx = box[0*3+0];
    RealType by = box[1*3+1];
    RealType bz = box[2*3+2];

    RealType inv_bx = 1/bx;
    RealType inv_by = 1/by;
    RealType inv_bz = 1/bz;

    RealType non_periodic_dist_i = 0;
    RealType non_periodic_dist_j = 0;

    // Determine if the row block can be translated into a periodic box
    // to optimize distance calculations
    // https://github.com/proteneer/timemachine/issues/320
    const bool single_periodic_box = (0.5f*bx-row_bb_ext_x >= cutoff &&
                                      0.5f*by-row_bb_ext_y >= cutoff &&
                                      0.5f*bz-row_bb_ext_z >= cutoff);
    if (single_periodic_box) {
        pos_i_x -= bx*nearbyint((pos_i_x-row_bb_ctr_x)*inv_bx);
        pos_i_y -= by*nearbyint((pos_i_y-row_bb_ctr_y)*inv_by);
        pos_i_z -= bz*nearbyint((pos_i_z-row_bb_ctr_z)*inv_bz);

        non_periodic_dist_i = static_cast<RealType>(0.5) * (pos_i_x*pos_i_x + pos_i_y*pos_i_y + pos_i_z*pos_i_z);
    }

    const RealType cutoff_squared = static_cast<RealType>(cutoff)*static_cast<RealType>(cutoff);

    int col_block_base = blockIdx.y*TILESIZE;

    int col_block_idx = col_block_base+indexInWarp;
    bool include_col_block = (col_block_idx < NUM_BLOCKS) && (col_block_idx >= row_block_idx);

    if (include_col_block) {

        // Compute center of column box and extent coords.
        RealType col_bb_ctr_x = bb_ctr[col_block_idx*3+0];
        RealType col_bb_ctr_y = bb_ctr[col_block_idx*3+1];
        RealType col_bb_ctr_z = bb_ctr[col_block_idx*3+2];

        RealType col_bb_ext_x = bb_ext[col_block_idx*3+0];
        RealType col_bb_ext_y = bb_ext[col_block_idx*3+1];
        RealType col_bb_ext_z = bb_ext[col_block_idx*3+2];

        // Find delta between boxes
        RealType box_box_dx = row_bb_ctr_x - col_bb_ctr_x;
        RealType box_box_dy = row_bb_ctr_y - col_bb_ctr_y;
        RealType box_box_dz = row_bb_ctr_z - col_bb_ctr_z;

        // Recenter delta box
        box_box_dx -= bx*nearbyint(box_box_dx*inv_bx);
        box_box_dy -= by*nearbyint(box_box_dy*inv_by);
        box_box_dz -= bz*nearbyint(box_box_dz*inv_bz);

        // If boxes overlap, treat distance as 0
        box_box_dx = max(static_cast<RealType>(0.0), fabs(box_box_dx) - row_bb_ext_x - col_bb_ext_x);
        box_box_dy = max(static_cast<RealType>(0.0), fabs(box_box_dy) - row_bb_ext_y - col_bb_ext_y);
        box_box_dz = max(static_cast<RealType>(0.0), fabs(box_box_dz) - row_bb_ext_z - col_bb_ext_z);

        // Check if the deltas between boxes are within cutoff
        include_col_block &= (box_box_dx*box_box_dx + box_box_dy*box_box_dy + box_box_dz*box_box_dz) < (cutoff_squared);

    }

    // __ballot returns bit flags to indicate which thread in the warp identified a column block within the cutoff.
    unsigned includeBlockFlags = __ballot_sync(FULL_MASK, include_col_block);

    // Loop over the col blocks we identified as potentially containing neighbors.
    while (includeBlockFlags != 0) {

        // (ytz): CUDA ffs returns an inclusive [0,32] such that:
        // ffs(0) == 0
        // ffs(2^0=1) == 1
        // ffs(2^1=2) == 2
        // ffs(2^2=4) == 3
        // ffs(2^3=8) == 4
        // ffs(2^31) == 32

        int offset = __ffs(includeBlockFlags)-1;

        includeBlockFlags &= includeBlockFlags-1;
        int col_block = col_block_base + offset;
        // Compute overlap between column bounding box and row atom
        RealType col_bb_ctr_x = bb_ctr[col_block*3+0];
        RealType col_bb_ctr_y = bb_ctr[col_block*3+1];
        RealType col_bb_ctr_z = bb_ctr[col_block*3+2];

        RealType col_bb_ext_x = bb_ext[col_block*3+0];
        RealType col_bb_ext_y = bb_ext[col_block*3+1];
        RealType col_bb_ext_z = bb_ext[col_block*3+2];

        // Don't use pos_i_* here, as might have been shifted to center
        RealType atom_box_dx = (atom_i_idx < N ? coords[atom_i_idx*3 + 0] : 0) - col_bb_ctr_x;
        RealType atom_box_dy = (atom_i_idx < N ? coords[atom_i_idx*3 + 1] : 0) - col_bb_ctr_y;
        RealType atom_box_dz = (atom_i_idx < N ? coords[atom_i_idx*3 + 2] : 0) - col_bb_ctr_z;

        atom_box_dx -= bx*nearbyint(atom_box_dx*inv_bx);
        atom_box_dy -= by*nearbyint(atom_box_dy*inv_by);
        atom_box_dz -= bz*nearbyint(atom_box_dz*inv_bz);

        atom_box_dx = max(static_cast<RealType>(0.0), fabs(atom_box_dx) - col_bb_ext_x);
        atom_box_dy = max(static_cast<RealType>(0.0), fabs(atom_box_dy) - col_bb_ext_y);
        atom_box_dz = max(static_cast<RealType>(0.0), fabs(atom_box_dz) - col_bb_ext_z);

        // Find rows where the row atom and column boxes are within cutoff
        bool interacts = false;
        unsigned atomFlags = __ballot_sync(FULL_MASK, atom_box_dx*atom_box_dx + atom_box_dy*atom_box_dy + atom_box_dz*atom_box_dz < cutoff_squared);
        int atom_j_idx = col_block*WARPSIZE+threadIdx.x; // each thread loads a different atom

        //       threadIdx
        //      0 1 2 3 4 5
        //   0  0 0 0 0 0 0 
        // a 1  0 1 0 1 1 0  row_atom
        // t 0  0 0 0 0 0 0
        // o 0  0 0 0 0 0 0 
        // m 0  0 0 0 0 0 0
        // f 1  1 0 0 0 1 1  row_atom
        // l 0  0 0 0 0 0 0
        // a 1  0 1 0 0 1 0  row_atom
        // g 1  1 1 0 0 0 1  row_atom
        // s 0  0 0 0 0 0 0
        //   0  0 0 0 0 0 0

        RealType pos_j_x = atom_j_idx < N ? coords[atom_j_idx*3 + 0] : 0;
        RealType pos_j_y = atom_j_idx < N ? coords[atom_j_idx*3 + 1] : 0;
        RealType pos_j_z = atom_j_idx < N ? coords[atom_j_idx*3 + 2] : 0;

        if (single_periodic_box) {
            // Recenter using **row** box center
            pos_j_x -= bx*nearbyint((pos_j_x-row_bb_ctr_x)*inv_bx);
            pos_j_y -= by*nearbyint((pos_j_y-row_bb_ctr_y)*inv_by);
            pos_j_z -= bz*nearbyint((pos_j_z-row_bb_ctr_z)*inv_bz);

            non_periodic_dist_j = static_cast<RealType>(0.5) * (pos_j_x*pos_j_x + pos_j_y*pos_j_y + pos_j_z*pos_j_z);
        }


        unsigned includeAtomFlags = 0;
        while(atomFlags) {
            const int row_atom = __ffs(atomFlags)-1;
            atomFlags &= atomFlags-1;
            RealType row_i_x = __shfl_sync(FULL_MASK, pos_i_x, row_atom);
            RealType row_i_y = __shfl_sync(FULL_MASK, pos_i_y, row_atom);
            RealType row_i_z = __shfl_sync(FULL_MASK, pos_i_z, row_atom);
            if (!single_periodic_box) {
                RealType atom_atom_dx = row_i_x - pos_j_x;
                RealType atom_atom_dy = row_i_y - pos_j_y;
                RealType atom_atom_dz = row_i_z - pos_j_z;

                atom_atom_dx -= bx*nearbyint(atom_atom_dx*inv_bx);
                atom_atom_dy -= by*nearbyint(atom_atom_dy*inv_by);
                atom_atom_dz -= bz*nearbyint(atom_atom_dz*inv_bz);

                interacts |= (atom_atom_dx*atom_atom_dx + atom_atom_dy*atom_atom_dy + atom_atom_dz*atom_atom_dz) < cutoff_squared;
            } else {
                // All threads in warp need single_periodic_box to be true for this not to hang
                RealType corrected_i = __shfl_sync(FULL_MASK, non_periodic_dist_i, row_atom);

                // Below is half the magnitude of the distance equation, expanded.
                RealType half_dist = corrected_i + non_periodic_dist_j - row_i_x*pos_j_x - row_i_y*pos_j_y - row_i_z*pos_j_z;
                interacts |= half_dist < (static_cast<RealType>(0.5) * cutoff_squared);
            }
            includeAtomFlags = __ballot_sync(FULL_MASK, interacts);
            // If all threads in the warp have found interactions, can terminate early
            if (includeAtomFlags == FULL_MASK) {
                break;
            }
        }
        
        // Add any interacting atoms to the buffer.
        if (interacts) {
            int index = neighborsInBuffer+__popc(includeAtomFlags & warpMask); // where to store this in shared memory
            // Indices can be at most 64
            ixn_j_buffer[index] = atom_j_idx;
        }
        neighborsInBuffer += __popc(includeAtomFlags);

        if(neighborsInBuffer > WARPSIZE) {
            int tilesToStore = 1;
            if(indexInWarp == 0) {
                sync_start[0] = atomicAdd(interactionCount, tilesToStore);
            }
            __syncwarp();
            interactingTiles[sync_start[0]] = row_block_idx;
            interactingAtoms[sync_start[0]*WARPSIZE + threadIdx.x] = ixn_j_buffer[threadIdx.x];

            ixn_j_buffer[threadIdx.x] = ixn_j_buffer[WARPSIZE+threadIdx.x];
            ixn_j_buffer[WARPSIZE+threadIdx.x] = N; // reset old values
            neighborsInBuffer -= WARPSIZE;
        }
    }
    
    // store trim
    const int Y = gridDim.y;
    trim_atoms[blockIdx.x*Y*WARPSIZE+blockIdx.y*WARPSIZE+threadIdx.x] = ixn_j_buffer[threadIdx.x];
}
