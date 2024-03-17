#include "k_sampling.cuh"
#include <assert.h>

namespace timemachine {

// References:
// https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
// https://lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/
template <typename RealType>
void __global__ k_setup_gumbel_max_trick(
    const int N,
    const RealType *__restrict__ log_weights,
    const RealType *__restrict__ gumbel_noise,
    RealType *__restrict__ prepared_gumbel) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) {
        return;
    }

    const RealType weight = log_weights[idx];
    assert(!isnan(weight));

    const RealType gumbel_rand = -log(-log(gumbel_noise[idx]));

    prepared_gumbel[idx] = weight + gumbel_rand;
}

template void __global__ k_setup_gumbel_max_trick<float>(
    const int N,
    const float *__restrict__ log_weights,
    const float *__restrict__ gumbel_noise,
    float *__restrict__ prepared_gumbel);

template void __global__ k_setup_gumbel_max_trick<double>(
    const int N,
    const double *__restrict__ log_weights,
    const double *__restrict__ gumbel_noise,
    double *__restrict__ prepared_gumbel);

template <typename RealType>
void __global__ k_setup_gumbel_max_trick_with_offset(
    const int num_segments,
    const int total_values,
    const int max_offset,
    const int *__restrict__ noise_offset,     // [1]
    const int *__restrict__ segment_offsets,  // [num_segments]
    const RealType *__restrict__ log_weights, // [total_values]
    const RealType *__restrict__ gumbel_noise,
    RealType *__restrict__ prepared_gumbel) {
    const int segment_idx = blockIdx.y;

    if (segment_idx >= num_segments) {
        return;
    }

    const int rand_offset = noise_offset[0];
    if (rand_offset + segment_idx >= max_offset) {
        return;
    }

    const int segment_start = segment_offsets[segment_idx];
    const int N = segment_offsets[segment_idx + 1] - segment_start;
    // In the case of the offset the values per segment need to match up to compute gumbel offset correctly.
    assert(N % total_values == 0);

    const int gumbel_offset = rand_offset * N + segment_start;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < N) {
        const RealType weight = log_weights[idx];
        assert(!isnan(weight));

        // If the idx in the batch segments is beyond the max offset return a negative infinity to avoid sampling the weight
        const RealType gumbel_rand = -log(-log(gumbel_noise[gumbel_offset + idx]));
        // printf("Idx %d - Idx with offset %d - %f\n", idx, rand_offset + idx, gumbel_rand);

        prepared_gumbel[segment_start + idx] = weight + gumbel_rand;

        idx += gridDim.x * blockDim.x;
    }
}

template void __global__ k_setup_gumbel_max_trick_with_offset<float>(
    const int num_segments,
    const int total_values,
    const int max_offset,
    const int *__restrict__ noise_offset,
    const int *__restrict__ segment_offsets, // [blockDim.y]
    const float *__restrict__ log_weights,
    const float *__restrict__ gumbel_noise,
    float *__restrict__ prepared_gumbel);

template void __global__ k_setup_gumbel_max_trick_with_offset<double>(
    const int num_segments,
    const int total_values,
    const int max_offset,
    const int *__restrict__ noise_offset,
    const int *__restrict__ segment_offsets, // [blockDim.y]
    const double *__restrict__ log_weights,
    const double *__restrict__ gumbel_noise,
    double *__restrict__ prepared_gumbel);

template <typename T>
void __global__
k_copy_kv_key(const int N, const cub::KeyValuePair<int, T> *__restrict__ kv_pairs, int *__restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) {
        return;
    }

    out[idx] = kv_pairs[idx].key;
}

template void __global__
k_copy_kv_key<float>(const int N, const cub::KeyValuePair<int, float> *__restrict__ kv_pairs, int *__restrict__ out);

template void __global__
k_copy_kv_key<double>(const int N, const cub::KeyValuePair<int, double> *__restrict__ kv_pairs, int *__restrict__ out);

} // namespace timemachine
