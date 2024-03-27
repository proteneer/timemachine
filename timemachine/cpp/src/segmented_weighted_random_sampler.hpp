#pragma once

#include <vector>

#include "curand.h"
#include "device_buffer.hpp"
#include <cub/util_type.cuh>

namespace timemachine {

// A SegmentedWeightedRandomSampler for random sampling from batches of probabilities (np.random.choice(X, replace=True p=probabilities))
// Uses a similar approach as the segmented APIs in cub
template <typename RealType> class SegmentedWeightedRandomSampler {

private:
    const int max_vals_per_segment_; // Max number of elements per segment
    const int num_segments_;         // Number of segments
    size_t temp_storage_bytes_;

    // Stores both the initial uniform random values and the final gumbel distribution
    DeviceBuffer<RealType> d_gumbel_;
    DeviceBuffer<cub::KeyValuePair<int, RealType>> d_arg_max_;

    DeviceBuffer<char> d_argmax_storage_;

    curandGenerator_t cr_rng_;

public:
    SegmentedWeightedRandomSampler(const int max_vals_per_segment, const int num_segments, const int seed);

    ~SegmentedWeightedRandomSampler();

    void sample_device(
        const int vals_per_segment,
        const int num_segments,
        const int *d_segment_offsets,
        const RealType *d_log_probabilities,
        int *d_samples,
        cudaStream_t stream);

    void sample_given_noise_device(
        const int vals_per_segment,
        const int num_segments,
        const int *d_segment_offsets,
        const RealType *d_log_probabilities,
        const RealType *d_noise,
        RealType *d_gumbel_dist, // Buffer to store the gumbel distribution
        int *d_samples,
        cudaStream_t stream);

    // Sometimes it is necessary to setup custom gumbel noise for the samples, such as in the case of TIBD where the noise
    // needs to be reused in a specific way to ensure bitwise determinism
    void sample_given_gumbel_noise_device(
        const int num_segments,
        const int *d_segment_offsets,   // [num_segments + 1]
        const RealType *d_gumbel_noise, // [total_values]
        int *d_samples,
        cudaStream_t stream);

    // sample_given_noise_and_offset_device is useful when sampling using a fixed pool of noise and the noise may need
    // to be reused by controlling the `d_noise_offset`.
    void sample_given_noise_and_offset_device(
        const int vals_per_segment,
        const int num_segments,
        const int max_offset,
        const int *d_segment_offsets,        // [num_segments]
        const RealType *d_log_probabilities, // [num_segments, vals_per_segment]
        const int *d_noise_offset,           // [num_segments, vals_per_segment]
        const RealType *d_noise,             // [num_segments, vals_per_segment]
        RealType *d_gumbel_noise,            // [num_segments, vals_per_segment] Buffer to store the gumbel distribution
        int *d_samples,                      // [num_segments]
        cudaStream_t stream);

    std::vector<int> sample_host(const std::vector<std::vector<RealType>> &probabilities);
};

} // namespace timemachine
