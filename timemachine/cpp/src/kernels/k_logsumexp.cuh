#pragma once

namespace timemachine {

// Convert the outputs of LogSumExp kernels into the final logsumexp value
template <typename RealType>
RealType __host__ __device__ __forceinline__ compute_logsumexp_final(const RealType max_val, const RealType sum) {
    return max_val + log(sum);
}

template <typename RealType>
void __global__ k_segmented_exp_sub_max(
    const int num_segments,
    const int *__restrict__ d_segment_offsets, // [num_segments]
    const RealType *__restrict__ max,          // [num_segments]
    const RealType *__restrict__ vals,         // [num_segments, K]
    RealType *__restrict__ out                 // [num_segments, K]
);

} // namespace timemachine
