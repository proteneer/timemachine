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

    void sum_device(
        const int total_values,
        const int num_segments,
        const int *d_segment_offsets,
        const RealType *d_values,
        RealType *d_max_out,
        RealType *d_exp_sum_out,
        cudaStream_t stream);

    std::vector<RealType> sum_host(std::vector<std::vector<RealType>> &vals);
};

} // namespace timemachine
