#include "k_logsumexp.cuh"

namespace timemachine {

// Convert the outputs of LogSumExp kernels into the final logsumexp value

template <typename RealType>
void __global__ k_exp_sub_max(
    const int N,
    const RealType *__restrict__ max,  // [1]
    const RealType *__restrict__ vals, // [N]
    RealType *__restrict__ out         // [N]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    out[idx] = exp(vals[idx] - max[0]);
}

template void __global__ k_exp_sub_max<float>(const int, const float *, const float *, float *);
template void __global__ k_exp_sub_max<double>(const int, const double *, const double *, double *);

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

template void __global__ k_segmented_exp_sub_max<float>(const int, const int *, const float *, const float *, float *);
template void __global__
k_segmented_exp_sub_max<double>(const int, const int *, const double *, const double *, double *);

} // namespace timemachine
