
#include "gpu_utils.cuh"
#include "kernels/k_logsumexp.cuh"
#include "kernels/kernel_utils.cuh"
#include "math_utils.cuh"
#include "segmented_logsumexp.hpp"
#include <cub/cub.cuh>

namespace timemachine {

template <typename RealType>
SegmentedLogSumExp<RealType>::SegmentedLogSumExp(const int max_vals_per_segment, const int num_segments)
    : max_vals_per_segment_(max_vals_per_segment), num_segments_(num_segments),
      d_temp_buffer_(max_vals_per_segment * num_segments), temp_storage_bytes_(0), d_temp_storage_buffer_(0) {
    void *dummy_temp = nullptr;
    RealType *dummy_in = nullptr;
    size_t max_storage_bytes = 0;
    cub::DeviceSegmentedReduce::Max(
        dummy_temp, max_storage_bytes, dummy_in, dummy_in, num_segments_, dummy_in, dummy_in);

    size_t sum_storage_bytes = 0;
    cub::DeviceSegmentedReduce::Sum(
        dummy_temp, sum_storage_bytes, dummy_in, dummy_in, num_segments_, dummy_in, dummy_in);

    // Allocate the larger of the two intermediate values, as we need to run both max and sum
    temp_storage_bytes_ = max(max_storage_bytes, sum_storage_bytes);
    d_temp_storage_buffer_.realloc(temp_storage_bytes_);
};

template <typename RealType> SegmentedLogSumExp<RealType>::~SegmentedLogSumExp(){};

template <typename RealType>
void SegmentedLogSumExp<RealType>::sum_device(
    const int total_values,
    const int num_segments,
    const int *d_segment_offsets,
    const RealType *d_values,
    RealType *d_max_sum_out,
    RealType *d_exp_sum_out,
    cudaStream_t stream) {
    if (total_values > max_vals_per_segment_ * num_segments_) {
        throw std::runtime_error(
            "total values is greater than buffer size:  total_values=" + std::to_string(total_values) +
            ", buffer_size=" + std::to_string(max_vals_per_segment_ * num_segments_));
    }
    if (num_segments != num_segments_) {
        throw std::runtime_error(
            "segments doesn't match: num_segments=" + std::to_string(num_segments) +
            ", num_segments_=" + std::to_string(num_segments));
    }
    const int tpb = DEFAULT_THREADS_PER_BLOCK;

    // Compute the max value, to subtract for the log computation
    gpuErrchk(cub::DeviceSegmentedReduce::Max(
        d_temp_storage_buffer_.data,
        temp_storage_bytes_,
        d_values,
        d_max_sum_out,
        num_segments,
        d_segment_offsets,
        d_segment_offsets + 1,
        stream));

    dim3 block(ceil_divide(total_values / num_segments, tpb), num_segments);
    k_segmented_exp_sub_max<<<block, tpb, 0, stream>>>(
        num_segments, d_segment_offsets, d_max_sum_out, d_values, d_temp_buffer_.data);
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cub::DeviceSegmentedReduce::Sum(
        d_temp_storage_buffer_.data,
        temp_storage_bytes_,
        d_temp_buffer_.data,
        d_exp_sum_out,
        num_segments,
        d_segment_offsets,
        d_segment_offsets + 1,
        stream));
}

template <typename RealType>
std::vector<RealType> SegmentedLogSumExp<RealType>::sum_host(std::vector<std::vector<RealType>> &vals) {
    const int num_segments = static_cast<int>(vals.size());
    std::vector<int> h_segments(num_segments + 1);

    int offset = 0;
    h_segments[0] = offset;
    std::vector<RealType> h_vals;
    for (unsigned long i = 0; i < num_segments; i++) {
        offset += vals[i].size();
        h_segments[i + 1] = offset;
        for (unsigned int j = 0; j < vals[i].size(); j++) {
            h_vals.push_back(vals[i][j]);
        }
    }

    DeviceBuffer<RealType> d_vals(h_vals);
    DeviceBuffer<RealType> d_max_out(num_segments);
    DeviceBuffer<RealType> d_sum_out(num_segments);
    DeviceBuffer<int> d_segment_offsets(h_segments);

    cudaStream_t stream = static_cast<cudaStream_t>(0);
    this->sum_device(
        h_segments[num_segments],
        num_segments,
        d_segment_offsets.data,
        d_vals.data,
        d_max_out.data,
        d_sum_out.data,
        stream);
    gpuErrchk(cudaStreamSynchronize(stream));

    std::vector<RealType> h_sum(num_segments);
    std::vector<RealType> h_max(num_segments);
    d_max_out.copy_to(&h_max[0]);
    d_sum_out.copy_to(&h_sum[0]);

    std::vector<RealType> h_logsumexp(num_segments);
    for (int i = 0; i < num_segments; i++) {
        h_logsumexp[i] = compute_logsumexp_final<RealType>(h_max[i], h_sum[i]);
    }
    return h_logsumexp;
};

template class SegmentedLogSumExp<double>;
template class SegmentedLogSumExp<float>;

} // namespace timemachine
