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
    const int *__restrict__ d_segment_offsets, // [num_segments + 1]
    const RealType *__restrict__ max,          // [num_segments]
    const RealType *__restrict__ vals,         // [num_segments, K]
    RealType *__restrict__ out                 // [num_segments, K]
) {
    int segment = blockIdx.y;
    while (segment < num_segments) {
        const int start = d_segment_offsets[segment];
        const int end = d_segment_offsets[segment + 1];
        const RealType max_val = max[segment];
        int idx = (blockIdx.x * blockDim.x + threadIdx.x) + start;
        while (idx < end) {
            out[idx] = exp(vals[idx] - max_val);

            idx += gridDim.x * blockDim.x;
        }
        segment += gridDim.y * blockDim.y;
    }
}

} // namespace timemachine
