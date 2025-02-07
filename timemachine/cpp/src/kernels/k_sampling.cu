#include "k_sampling.cuh"

namespace timemachine {

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

template void __global__ k_setup_gumbel_max_trick_targeted_insertion<float>(
    const int num_segments,
    const int num_noise_per_segment,
    const int max_offset,
    const int *__restrict__ noise_offset,
    const int *__restrict__ segment_offsets, // [blockDim.y]
    const float *__restrict__ log_weights,
    const float *__restrict__ gumbel_noise,
    float *__restrict__ prepared_gumbel);

template void __global__ k_setup_gumbel_max_trick_targeted_insertion<double>(
    const int num_segments,
    const int num_noise_per_segment,
    const int max_offset,
    const int *__restrict__ noise_offset,
    const int *__restrict__ segment_offsets, // [blockDim.y]
    const double *__restrict__ log_weights,
    const double *__restrict__ gumbel_noise,
    double *__restrict__ prepared_gumbel);

template void __global__
k_copy_kv_key<float>(const int N, const cub::KeyValuePair<int, float> *__restrict__ kv_pairs, int *__restrict__ out);

template void __global__
k_copy_kv_key<double>(const int N, const cub::KeyValuePair<int, double> *__restrict__ kv_pairs, int *__restrict__ out);

} // namespace timemachine
