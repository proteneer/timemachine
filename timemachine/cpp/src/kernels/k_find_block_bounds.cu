#include <cstdio>

#define WARPSIZE 32


// /**
//  * Compare the bounding boxes for each pair of atom blocks (comprised of 32 atoms each), forming a tile. If the two
//  * atom blocks are sufficiently far apart, mark them as non-interacting. There are two stages in the algorithm.
//  *
//  * STAGE 1:
//  *
//  * A coarse grained atom block against interacting atom block neighbour list is constructed. 
//  *
//  * Each warp first loads in some block X of interest. Each thread within the warp then loads 
//  * in a different atom block Y. If Y has exclusions with X, then Y is not processed.  If the bounding boxes 
//  * of the two atom blocks are within the cutoff distance, then the two atom blocks are considered to be
//  * interacting and Y is added to the buffer for X.
//  *
//  * STAGE 2:
//  *
//  * A fine grained atom block against interacting atoms neighbour list is constructed.
//  *
//  * The warp loops over atom blocks Y that were found to (possibly) interact with atom block X.  Each thread
//  * in the warp loops over the 32 atoms in X and compares their positions to one particular atom from block Y.
//  * If it finds one closer than the cutoff distance, the atom is added to the list of atoms interacting with block X.
//  * This continues until the buffer fills up, at which point the results are written to global memory.
//  *
//  * [in] periodicBoxSize        - size of the rectangular periodic box
//  * [in] invPeriodicBoxSize     - inverse of the periodic box
//  * [in] blockCenter            - the center of each bounding box
//  * [in] blockBoundingBox       - bounding box of each atom block
//  * [out] interactionCount      - total number of tiles that have interactions
//  * [out] interactingTiles      - set of blocks that have interactions
//  * [out] interactingAtoms      - a list of atoms that interact with each atom block
//  * [in] posq                   - x,y,z coordinates of each atom and charge q
//  * [in] maxTiles               - maximum number of tiles to process, used for multi-GPUs
//  * [in] startBlockIndex        - first block to process, used for multi-GPUs,
//  * [in] numBlocks              - total number of atom blocks
//  * [in] sortedBlocks           - a sorted list of atom blocks based on volume
//  * [in] sortedBlockCenter      - sorted centers, duplicated for fast access to avoid indexing
//  * [in] sortedBlockBoundingBox - sorted bounding boxes, duplicated for fast access
//  * [in] exclusionIndices       - maps into exclusionRowIndices with the starting position for a given atom
//  * [in] exclusionRowIndices    - stores the a continuous list of exclusions
//  *           eg: block 0 is excluded from atom 3,5,6
//  *               block 1 is excluded from atom 3,4
//  *               block 2 is excluded from atom 1,3,5,6
//  *              exclusionIndices[0][3][5][8]
//  *           exclusionRowIndices[3][5][6][3][4][1][3][5][6]
//  *                         index 0  1  2  3  4  5  6  7  8 
//  * [out] oldPos                - stores the positions of the atoms in which this neighbourlist was built on
//  *                             - this is used to decide when to rebuild a neighbourlist
//  * [in] rebuildNeighbourList   - whether or not to execute this kernel
//  *
//  */



    // If we have a partially filled buffer,  store it to memory.
    
// #if MAX_BITS_FOR_PAIRS > 0
//     if (neighborsInBuffer > 32)
//         neighborsInBuffer = saveSinglePairs(x, buffer, flagsBuffer, neighborsInBuffer, maxSinglePairs, &interactionCount[1], singlePairs, sumBuffer+warpStart, pairStartIndex);
// #endif
    // if (neighborsInBuffer > 0) {
    //     int tilesToStore = (neighborsInBuffer+TILE_SIZE-1)/TILE_SIZE;
    //     if (indexInWarp == 0)
    //         tileStartIndex = atomicAdd(&interactionCount[0], tilesToStore);
    //     int newTileStartIndex = tileStartIndex;
    //     if (newTileStartIndex+tilesToStore <= maxTiles) {
    //         if (indexInWarp < tilesToStore)
    //             interactingTiles[newTileStartIndex+indexInWarp] = x;
    //         for (int j = 0; j < tilesToStore; j++)
    //             interactingAtoms[(newTileStartIndex+j)*TILE_SIZE+indexInWarp] = (indexInWarp+j*TILE_SIZE < neighborsInBuffer ? buffer[indexInWarp+j*TILE_SIZE] : NUM_ATOMS);
    //     }
    // }
    // }
    
    // Record the positions the neighbor list is based on.
    
    // for (int i = threadIdx.x+blockIdx.x*blockDim.x; i < NUM_ATOMS; i += blockDim.x*gridDim.x)
        // oldPositions[i] = posq[i];
// }



void __global__ k_find_block_bounds(
    const int N,
    const int D,
    const int T,
    const double *coords,
    const double *box,
    const int *perm,
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
                int atom_idx = perm[tid];
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
