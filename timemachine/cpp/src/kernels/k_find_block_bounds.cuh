#pragma once

#define BYTE_TO_BINARY_PATTERN "%c%c%c%c%c%c%c%c"
#define BYTE_TO_BINARY(byte)  \
  (byte & 0x80 ? '1' : '0'), \
  (byte & 0x40 ? '1' : '0'), \
  (byte & 0x20 ? '1' : '0'), \
  (byte & 0x10 ? '1' : '0'), \
  (byte & 0x08 ? '1' : '0'), \
  (byte & 0x04 ? '1' : '0'), \
  (byte & 0x02 ? '1' : '0'), \
  (byte & 0x01 ? '1' : '0') 

void __global__ k_find_block_bounds(
    const int N,
    const int D,
    const int T,
    const double *coords,
    const double *box,
    // const int *perm,
    double *block_bounds_ctr,
    double *block_bounds_ext);


template<typename RealType>
void __global__ find_blocks_with_interactions(
    int N,
    const double *bb_ctr,
    const double *bb_ext,
    const double* __restrict__ coords,
    const double* __restrict__ box,
    unsigned int* __restrict__ interactionCount,
    int* __restrict__ interactingTiles,
    unsigned int* __restrict__ interactingAtoms,
    // real4* __restrict__ oldPositions,
    // const int* __restrict__ rebuildNeighborList) {
    double cutoff) {

    const int indexInWarp = threadIdx.x%32;
    const int warpMask = (1<<indexInWarp)-1;
    __shared__ RealType pos_i_buffer[32*3];
    __shared__ int ixn_j_buffer[64]; // we can probably get away with using only 32 if we do some fancier remainder tricks

    // initialize
    ixn_j_buffer[threadIdx.x] = N;
    ixn_j_buffer[32+threadIdx.x] = N;

    __shared__ volatile int sync_start[1];
    
    int startBlockIndex = 0;
    int row_block_idx = blockIdx.x;

    RealType row_bb_ctr_x = bb_ctr[row_block_idx*3+0];
    RealType row_bb_ctr_y = bb_ctr[row_block_idx*3+1];
    RealType row_bb_ctr_z = bb_ctr[row_block_idx*3+2];

    RealType row_bb_ext_x = bb_ext[row_block_idx*3+0];
    RealType row_bb_ext_y = bb_ext[row_block_idx*3+1];
    RealType row_bb_ext_z = bb_ext[row_block_idx*3+2];


    int neighborsInBuffer = 0;

    unsigned int atom_i_idx = blockIdx.x*blockDim.x + threadIdx.x;

    RealType pos_i_x = atom_i_idx < N ? coords[atom_i_idx*3 + 0] : 0;
    RealType pos_i_y = atom_i_idx < N ? coords[atom_i_idx*3 + 1] : 0;
    RealType pos_i_z = atom_i_idx < N ? coords[atom_i_idx*3 + 2] : 0;

    pos_i_buffer[threadIdx.x*3 + 0] = pos_i_x;
    pos_i_buffer[threadIdx.x*3 + 1] = pos_i_y;
    pos_i_buffer[threadIdx.x*3 + 2] = pos_i_z;

    const int NUM_BLOCKS = (N+32-1)/32;

    RealType bx = box[0*3+0];
    RealType by = box[1*3+1];
    RealType bz = box[2*3+2];

    RealType cutoff_squared = static_cast<RealType>(cutoff)*static_cast<RealType>(cutoff);

    // int col_block_start = max(row_block_idx, blockIdx.y);
    // int col_block_end = blockIdx.y+1

    for (int col_block_base = row_block_idx; col_block_base < NUM_BLOCKS; col_block_base += 32) {

        int col_block_idx = col_block_base+indexInWarp;
        bool include_col_block = (col_block_idx < NUM_BLOCKS);

        if (include_col_block) {

            RealType col_bb_ctr_x = bb_ctr[col_block_idx*3+0];
            RealType col_bb_ctr_y = bb_ctr[col_block_idx*3+1];
            RealType col_bb_ctr_z = bb_ctr[col_block_idx*3+2];

            RealType col_bb_ext_x = bb_ext[col_block_idx*3+0];
            RealType col_bb_ext_y = bb_ext[col_block_idx*3+1];
            RealType col_bb_ext_z = bb_ext[col_block_idx*3+2];

            RealType box_box_dx = row_bb_ctr_x - col_bb_ctr_x;
            RealType box_box_dy = row_bb_ctr_y - col_bb_ctr_y;
            RealType box_box_dz = row_bb_ctr_z - col_bb_ctr_z;

            box_box_dx -= bx*floor(box_box_dx/bx + static_cast<RealType>(0.5));
            box_box_dy -= by*floor(box_box_dy/by + static_cast<RealType>(0.5));
            box_box_dz -= bz*floor(box_box_dz/bz + static_cast<RealType>(0.5));

            box_box_dx = max(static_cast<RealType>(0.0), fabs(box_box_dx) - row_bb_ext_x - col_bb_ext_x);
            box_box_dy = max(static_cast<RealType>(0.0), fabs(box_box_dy) - row_bb_ext_y - col_bb_ext_y);
            box_box_dz = max(static_cast<RealType>(0.0), fabs(box_box_dz) - row_bb_ext_z - col_bb_ext_z);

            include_col_block &= (box_box_dx*box_box_dx + box_box_dy*box_box_dy + box_box_dz*box_box_dz) < cutoff_squared;

        }


        // __ballot returns number of col blocks in this group whose distance to the row block is closer than the cutoff.
        unsigned includeBlockFlags = __ballot_sync(0xffffffff, include_col_block);

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

            RealType atom_box_dx = pos_i_buffer[threadIdx.x*3 + 0] - col_bb_ctr_x;
            RealType atom_box_dy = pos_i_buffer[threadIdx.x*3 + 1] - col_bb_ctr_y;
            RealType atom_box_dz = pos_i_buffer[threadIdx.x*3 + 2] - col_bb_ctr_z;

            atom_box_dx -= bx*floor(atom_box_dx/bx + static_cast<RealType>(0.5));
            atom_box_dy -= by*floor(atom_box_dy/by + static_cast<RealType>(0.5));
            atom_box_dz -= bz*floor(atom_box_dz/bz + static_cast<RealType>(0.5));

            atom_box_dx = max(static_cast<RealType>(0.0), fabs(atom_box_dx) - col_bb_ext_x);
            atom_box_dy = max(static_cast<RealType>(0.0), fabs(atom_box_dy) - col_bb_ext_y);
            atom_box_dz = max(static_cast<RealType>(0.0), fabs(atom_box_dz) - col_bb_ext_z);

            // // every atom should be in the bounding box
            // printf("atom_i_idx %d row_block %d deltas %f %f %f pass? %d \n", atom_i_idx, row_block_idx, atom_box_dx, atom_box_dy, atom_box_dz, sqrt(atom_box_dx*atom_box_dx + atom_box_dy*atom_box_dy + atom_box_dz*atom_box_dz) < cutoff);

            unsigned atomFlags = __ballot_sync(0xffffffff, atom_box_dx*atom_box_dx + atom_box_dy*atom_box_dy + atom_box_dz*atom_box_dz < cutoff_squared);
            int interacts = 0;
            int atom_j_idx = col_block*32+threadIdx.x; // each thread loads a different atom2

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

            while(atomFlags) {

                    RealType pos_j_x = atom_j_idx < N ? coords[atom_j_idx*3 + 0] : 0;
                    RealType pos_j_y = atom_j_idx < N ? coords[atom_j_idx*3 + 1] : 0;
                    RealType pos_j_z = atom_j_idx < N ? coords[atom_j_idx*3 + 2] : 0;

                    int row_atom = __ffs(atomFlags)-1;
                    atomFlags &= atomFlags-1;

                    RealType atom_atom_dx = pos_i_buffer[row_atom*3 + 0] - pos_j_x;
                    RealType atom_atom_dy = pos_i_buffer[row_atom*3 + 1] - pos_j_y;
                    RealType atom_atom_dz = pos_i_buffer[row_atom*3 + 2] - pos_j_z;

                    atom_atom_dx -= bx*floor(atom_atom_dx/bx + static_cast<RealType>(0.5));
                    atom_atom_dy -= by*floor(atom_atom_dy/by + static_cast<RealType>(0.5));
                    atom_atom_dz -= bz*floor(atom_atom_dz/bz + static_cast<RealType>(0.5));

                    interacts |= (atom_atom_dx*atom_atom_dx + atom_atom_dy*atom_atom_dy + atom_atom_dz*atom_atom_dz < cutoff_squared ? 1<<row_atom : 0);

            }
            
            // Add any interacting atoms to the buffer.
            
            int includeAtomFlags = __ballot_sync(0xffffffff, interacts);
            if (interacts) {
                int index = neighborsInBuffer+__popc(includeAtomFlags & warpMask); // where to store this in shared memory
                ixn_j_buffer[index] = atom_j_idx; // store this atom into buffer only if it passes the distance test
                // flagsBuffer[index] = interacts;
            }
            neighborsInBuffer += __popc(includeAtomFlags);

            // tiles to store == 1
            if(neighborsInBuffer > 32) {
                int tilesToStore = 1;
                if(indexInWarp == 0) {
                    sync_start[0] = atomicAdd(interactionCount, tilesToStore);
                }
                interactingTiles[sync_start[0]] = row_block_idx;
                interactingAtoms[sync_start[0]*32 + threadIdx.x] = ixn_j_buffer[threadIdx.x];

                ixn_j_buffer[threadIdx.x] = ixn_j_buffer[32+threadIdx.x];
                ixn_j_buffer[32+threadIdx.x] = N; // reset old values
                neighborsInBuffer -= 32;
            }
        } // loop over row atoms
    } // loop over all col blocks

    if(neighborsInBuffer > 0) {
        int tilesToStore = 1;
        if(indexInWarp == 0) {
            sync_start[0] = atomicAdd(interactionCount, tilesToStore);
        }
        interactingTiles[sync_start[0]] = row_block_idx;
        interactingAtoms[sync_start[0]*32 + threadIdx.x] = ixn_j_buffer[threadIdx.x];
    }
}
    