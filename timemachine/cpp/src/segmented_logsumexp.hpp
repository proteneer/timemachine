#pragma once

#include "device_buffer.hpp"

namespace timemachine {

template <typename RealType> class SegmentedLogSumExp {

private:
    const int max_vals_per_segment_;
    const int num_segments_;
    DeviceBuffer<RealType> d_temp_buffer_;
    // Buffers for finding the max/summed value
    std::size_t temp_storage_bytes_;
    DeviceBuffer<char> d_temp_storage_buffer_;

public:
    SegmentedLogSumExp(const int max_vals_per_segment, const int num_segments);

    ~SegmentedLogSumExp();

    // sum_device stores the max value of each segment as well as the sum(exp(x - max) for x in xs)
    // in two separate arrays. This is done to avoid having to run an additional kernel to combine the values
    // when it is trivial to do in a downstream kernel consuming the logsumexp.
    void sum_device(
        const int total_values,
        const int num_segments,
        const int *d_segment_offsets,
        const RealType *d_values,
        RealType *d_max_out,
        RealType *d_exp_sum_out,
        cudaStream_t stream);

    // sum_host returns a vector of the final logsumexp value of a set of segments.
    // This code combines the max value and the exp_sum that values are written to by sum_device
    std::vector<RealType> sum_host(std::vector<std::vector<RealType>> &vals);
};

} // namespace timemachine
