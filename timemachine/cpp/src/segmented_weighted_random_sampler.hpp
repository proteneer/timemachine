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
        const int total_values,
        const int num_segments,
        const int *d_segment_offsets,
        const RealType *d_log_probabilities,
        int *d_samples,
        cudaStream_t stream);

    void sample_given_noise_device(
        const int total_values,
        const int num_segments,
        const int *d_segment_offsets,
        const RealType *d_log_probabilities,
        const RealType *d_noise,
        RealType *d_gumbel_dist, // Buffer to store the gumbel distribution
        int *d_samples,
        cudaStream_t stream);

    std::vector<int> sample_host(const std::vector<std::vector<RealType>> &probabilities);
};

} // namespace timemachine
