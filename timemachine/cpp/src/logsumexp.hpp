#pragma once

#include "device_buffer.hpp"

namespace timemachine {

template <typename RealType> class LogSumExp {

private:
    const int N_;
    const DeviceBuffer<RealType> d_temp_buffer_;
    // Buffers for finding the max/summed value
    std::size_t temp_storage_bytes_;
    DeviceBuffer<char> d_temp_storage_buffer_;

public:
    LogSumExp(const int N);

    ~LogSumExp();

    void sum_device(const int N, const RealType *d_values, RealType *d_exp_sum_out, cudaStream_t stream);

    void sum_host(const int N, const RealType *h_values, RealType *h_out);
};

} // namespace timemachine
