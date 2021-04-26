#pragma once

#define FULL_MASK 0xffffffff


void __global__ k_find_block_bounds(
    const int N,
    const int D,
    const int T,
    const double *coords,
    const double *box,
    double *block_bounds_ctr,
    double *block_bounds_ext);

void __global__ k_compact_trim_atoms(
    int N,
    int Y,
    unsigned int* __restrict__ trim_atoms,
    unsigned int* __restrict__ interactionCount,
    int* __restrict__ interactingTiles,
    unsigned int* __restrict__ interactingAtoms) {

    __shared__ int ixn_j_buffer[64]; // we can probably get away with using only 32 if we do some fancier remainder tricks, but this isn't a huge save
    ixn_j_buffer[threadIdx.x] = N;
    ixn_j_buffer[32+threadIdx.x] = N;

    const int indexInWarp = threadIdx.x%32;
    const int warpMask = (1<<indexInWarp)-1;
    const int row_block_idx = blockIdx.x;

    __shared__ volatile int sync_start[1];
    int neighborsInBuffer = 0;

    for(int trim_block_idx=0; trim_block_idx < Y; trim_block_idx++) {

        int atom_j_idx = trim_atoms[row_block_idx*Y*32 + trim_block_idx*32 + threadIdx.x];
        bool interacts = atom_j_idx < N;

        int includeAtomFlags = __ballot_sync(FULL_MASK, interacts);

        if (interacts) {
            // only interacting atoms partake in this
            int index = neighborsInBuffer+__popc(includeAtomFlags & warpMask); // where to store this in shared memory
            ixn_j_buffer[index] = atom_j_idx;
        }
        neighborsInBuffer += __popc(includeAtomFlags);

        if(neighborsInBuffer > 32) {
            int tilesToStore = 1;
            if(indexInWarp == 0) {
                sync_start[0] = atomicAdd(interactionCount, tilesToStore);
            }
            __syncwarp();
            interactingTiles[sync_start[0]] = row_block_idx; // IS THIS CORRECT? CONTESTED
            interactingAtoms[sync_start[0]*32 + threadIdx.x] = ixn_j_buffer[threadIdx.x];

            ixn_j_buffer[threadIdx.x] = ixn_j_buffer[32+threadIdx.x];
            ixn_j_buffer[32+threadIdx.x] = N; // reset old values
            neighborsInBuffer -= 32;
        }
    }

    if(neighborsInBuffer > 0) {
        int tilesToStore = 1;
        if(indexInWarp == 0) {
            sync_start[0] = atomicAdd(interactionCount, tilesToStore);
        }
        __syncwarp();
        interactingTiles[sync_start[0]] = row_block_idx;
        interactingAtoms[sync_start[0]*32 + threadIdx.x] = ixn_j_buffer[threadIdx.x];
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
    int N,
    const double *bb_ctr,
    const double *bb_ext,
    const double* __restrict__ coords, //TBD make float32 version
    const double* __restrict__ box,
    unsigned int* __restrict__ interactionCount, // number of tiles that have interactions
    int* __restrict__ interactingTiles, // the row block idx of the tile that is interacting
    unsigned int* __restrict__ interactingAtoms, // the col block of the atoms that are interacting
    unsigned int* __restrict__ trim_atoms, // the left-over trims that will later be compacted
    double cutoff) {

    const int indexInWarp = threadIdx.x%32;
    const int warpMask = (1<<indexInWarp)-1;

    __shared__ int ixn_j_buffer[64]; // we can probably get away with using only 32 if we do some fancier remainder tricks, but this isn't a huge save

    // initialize
    ixn_j_buffer[threadIdx.x] = N;
    ixn_j_buffer[32+threadIdx.x] = N;

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

    const int NUM_BLOCKS = (N+32-1)/32;

    RealType bx = box[0*3+0];
    RealType by = box[1*3+1];
    RealType bz = box[2*3+2];


    RealType inv_bx = 1/bx;
    RealType inv_by = 1/by;
    RealType inv_bz = 1/bz;

    RealType cutoff_squared = static_cast<RealType>(cutoff)*static_cast<RealType>(cutoff);

    int col_block_base = blockIdx.y*32;

    int col_block_idx = col_block_base+indexInWarp;
    bool include_col_block = (col_block_idx < NUM_BLOCKS) && (col_block_idx >= row_block_idx);

    if (include_col_block) {

        // Compute center of column box and external coords. 
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

        box_box_dx -= bx*nearbyint(box_box_dx*inv_bx);
        box_box_dy -= by*nearbyint(box_box_dy*inv_by);
        box_box_dz -= bz*nearbyint(box_box_dz*inv_bz);

        box_box_dx = max(static_cast<RealType>(0.0), fabs(box_box_dx) - row_bb_ext_x - col_bb_ext_x);
        box_box_dy = max(static_cast<RealType>(0.0), fabs(box_box_dy) - row_bb_ext_y - col_bb_ext_y);
        box_box_dz = max(static_cast<RealType>(0.0), fabs(box_box_dz) - row_bb_ext_z - col_bb_ext_z);

        // Check if the deltas between boxes are within cutoff
        include_col_block &= (box_box_dx*box_box_dx + box_box_dy*box_box_dy + box_box_dz*box_box_dz) < cutoff_squared;

    }

    // __ballot returns number of col blocks in this group whose distance to the row block is closer than the cutoff.
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

        RealType col_bb_ctr_x = bb_ctr[col_block*3+0];
        RealType col_bb_ctr_y = bb_ctr[col_block*3+1];
        RealType col_bb_ctr_z = bb_ctr[col_block*3+2];

        RealType col_bb_ext_x = bb_ext[col_block*3+0];
        RealType col_bb_ext_y = bb_ext[col_block*3+1];
        RealType col_bb_ext_z = bb_ext[col_block*3+2];

        RealType atom_box_dx = pos_i_x - col_bb_ctr_x;
        RealType atom_box_dy = pos_i_y - col_bb_ctr_y;
        RealType atom_box_dz = pos_i_z - col_bb_ctr_z;

        atom_box_dx -= bx*nearbyint(atom_box_dx*inv_bx);
        atom_box_dy -= by*nearbyint(atom_box_dy*inv_by);
        atom_box_dz -= bz*nearbyint(atom_box_dz*inv_bz);

        atom_box_dx = max(static_cast<RealType>(0.0), fabs(atom_box_dx) - col_bb_ext_x);
        atom_box_dy = max(static_cast<RealType>(0.0), fabs(atom_box_dy) - col_bb_ext_y);
        atom_box_dz = max(static_cast<RealType>(0.0), fabs(atom_box_dz) - col_bb_ext_z);

        // Find atoms where the row and column boxes within cutoff
        unsigned atomFlags = __ballot_sync(FULL_MASK, atom_box_dx*atom_box_dx + atom_box_dy*atom_box_dy + atom_box_dz*atom_box_dz < cutoff_squared);
        int interacts = 0;
        int atom_j_idx = col_block*32+threadIdx.x; // each thread loads a different atom

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

        while(atomFlags) {

            int row_atom = __ffs(atomFlags)-1;
            atomFlags &= atomFlags-1;

            RealType atom_atom_dx = __shfl_sync(FULL_MASK, pos_i_x, row_atom) - pos_j_x;
            RealType atom_atom_dy = __shfl_sync(FULL_MASK, pos_i_y, row_atom) - pos_j_y;
            RealType atom_atom_dz = __shfl_sync(FULL_MASK, pos_i_z, row_atom) - pos_j_z;

            atom_atom_dx -= bx*nearbyint(atom_atom_dx*inv_bx);
            atom_atom_dy -= by*nearbyint(atom_atom_dy*inv_by);
            atom_atom_dz -= bz*nearbyint(atom_atom_dz*inv_bz);
            interacts |= (atom_atom_dx*atom_atom_dx + atom_atom_dy*atom_atom_dy + atom_atom_dz*atom_atom_dz < cutoff_squared ? 1<<row_atom : 0);
        }
        
        // Add any interacting atoms to the buffer.
        int includeAtomFlags = __ballot_sync(FULL_MASK, interacts);

        if (interacts) {
            int index = neighborsInBuffer+__popc(includeAtomFlags & warpMask); // where to store this in shared memory
            // Indices can be at most 64
            ixn_j_buffer[index] = atom_j_idx;
        }
        neighborsInBuffer += __popc(includeAtomFlags);

        if(neighborsInBuffer > 32) {
            int tilesToStore = 1;
            if(indexInWarp == 0) {
                sync_start[0] = atomicAdd(interactionCount, tilesToStore);
            }
            __syncwarp();
            interactingTiles[sync_start[0]] = row_block_idx;
            interactingAtoms[sync_start[0]*32 + threadIdx.x] = ixn_j_buffer[threadIdx.x];

            ixn_j_buffer[threadIdx.x] = ixn_j_buffer[32+threadIdx.x];
            ixn_j_buffer[32+threadIdx.x] = N; // reset old values
            neighborsInBuffer -= 32;
        }
    }
    
    // store trim
    int Y = gridDim.y;
    trim_atoms[blockIdx.x*Y*32+blockIdx.y*32+threadIdx.x] = ixn_j_buffer[threadIdx.x];

}
