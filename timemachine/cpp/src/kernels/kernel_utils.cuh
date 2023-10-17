#pragma once

namespace timemachine {

// box cache is a struct that can be used as shared memory in kernels to reduce
// global memory reads and to reduce the amount of work done per block.
template <typename RealType> struct box_cache {
    RealType x;
    RealType y;
    RealType z;
    RealType inv_x;
    RealType inv_y;
    RealType inv_z;
};

// DEFAULT_THREADS_PER_BLOCK should be at least 128 to ensure maximum occupancy for Cuda Arch 8.6 with 48 SMs
// given that there aren't too many registers in the kernel.
// Refer to the occupancy calculator in Nsight Compute for more details
static const int DEFAULT_THREADS_PER_BLOCK = 128;
static const int WARP_SIZE = 32;
// DEFAULT_THREADS_PER_BLOCK should be multiple of WARP_SIZE, else it is wasteful
static_assert(DEFAULT_THREADS_PER_BLOCK % WARP_SIZE == 0);

} // namespace timemachine
