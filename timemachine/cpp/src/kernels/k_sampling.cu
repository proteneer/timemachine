#include "k_sampling.cuh"
#include <assert.h>
#include <cmath>

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

    const RealType log_weight = log_weights[idx];
    // -inf is alright since that is log(0.0), +inf is not
    assert(!isnan(log_weight) && log_weight != INFINITY);

    const RealType gumbel_rand = -log(-log(gumbel_noise[idx]));

    prepared_gumbel[idx] = log_weight + gumbel_rand;
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
    assert(total_values % N == 0);

    const int gumbel_offset = rand_offset * N + segment_start;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < N) {
        const RealType log_weight = log_weights[idx];
        // -inf is alright since that is log(0.0), +inf is not
        assert(!isnan(log_weight) && log_weight != INFINITY);

        const RealType gumbel_rand = -log(-log(gumbel_noise[gumbel_offset + idx]));

        prepared_gumbel[segment_start + idx] = log_weight + gumbel_rand;

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

/* k_setup_gumbel_max_trick_targeted_insertion is a specialized version of
* k_setup_gumbel_max_trick_with_offset to handle the fact that we generate num_noise_per_segment values of noise
* per segment to ensure that we get bitwise deterministic results. If k_setup_gumbel_max_trick_with_offset
* were to be used, the data would be re-used but you wouldn't be able to get the same results depending on
* the number of proposals per move.
*
* Example of why not use k_setup_gumbel_max_trick_with_offset:
*
* num_noise_per_segment = 4
* inner_count = 3
* outer_count = 1
*
* If the proposals_per_move were 1, and had two iterations the results would look like:
* Iteration 1
*     Move targeting inner (evaluating the single outer molecule)
*     noise = [a, b, c, d]
*              ^ Only use the first value
*     weights = [1]
*     Sampler -> 0 // First idx
* Iteration 2
*     Move targeting outer (evaluating the three inner molecules)
*     noise = [e, f, g, h]
*              ^ Looking at e rather than b for the noise
*     weights = [2, 3, 4]
*     Sampler -> 1 // Second idx
*
* If the proposals_per_move were 2, and had one iteration the results would look like:
* Iteration 1
*     First targeting inner (evaluating the single outer molecule) then outer (three mols)
*     noise = [a, b, c, d]
*                 ^ Used b noise for 2 rather than e in the case of a single proposal per move
*     weights = [[1], [2, 3, 4]]
*     Sampler -> [0, 0] // First idxs of both segments selected, rather than first and second idx.
*
* This kernel forces such that each set of weights always uses the full num_noise_per_segment, avoiding
* the difference between proposals_per_move.
*/
template <typename RealType>
void __global__ k_setup_gumbel_max_trick_targeted_insertion(
    const int num_segments,
    const int num_noise_per_segment,
    const int max_offset,
    const int *__restrict__ noise_offset,      // [1]
    const int *__restrict__ segment_offsets,   // [num_segments]
    const RealType *__restrict__ log_weights,  // [total_values]
    const RealType *__restrict__ gumbel_noise, // [max_offset]
    RealType *__restrict__ prepared_gumbel     // [total_values]
) {
    const int segment_idx = blockIdx.y;

    if (segment_idx >= num_segments) {
        return;
    }

    const int rand_offset = noise_offset[0];

    const int gumbel_offset = rand_offset * num_noise_per_segment + segment_idx * num_noise_per_segment;
    if (gumbel_offset + num_noise_per_segment > max_offset) {
        return;
    }
    const int segment_start = segment_offsets[segment_idx];
    const int segment_end = segment_offsets[segment_idx + 1];
    const int N = segment_end - segment_start;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < N) {
        const RealType log_weight = log_weights[segment_start + idx];
        // -inf is alright since that is log(0.0), +inf is not
        assert(!isnan(log_weight) && log_weight != INFINITY);

        const RealType gumbel_rand = -log(-log(gumbel_noise[gumbel_offset + idx]));

        prepared_gumbel[segment_start + idx] = log_weight + gumbel_rand;
        idx += gridDim.x * blockDim.x;
    }
}

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

} // namespace timemachine
