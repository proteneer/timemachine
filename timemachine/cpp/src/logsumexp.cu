
#include "gpu_utils.cuh"
#include "kernels/k_logsumexp.cuh"
#include "kernels/kernel_utils.cuh"
#include "logsumexp.hpp"
#include "math_utils.cuh"
#include <cub/cub.cuh>

namespace timemachine {

template <typename RealType>
LogSumExp<RealType>::LogSumExp(const int N)
    : N_(N), d_temp_buffer_(N), temp_storage_bytes_(0), d_temp_storage_buffer_(0) {
    void *dummy_temp = nullptr;
    RealType *dummy_in = nullptr;
    size_t max_storage_bytes = 0;
    cub::DeviceReduce::Max(dummy_temp, max_storage_bytes, dummy_in, dummy_in, N_);

    size_t sum_storage_bytes = 0;
    cub::DeviceReduce::Sum(dummy_temp, sum_storage_bytes, dummy_in, dummy_in, N_);

    // Allocate the larger of the two intermediate values, as we need to run both max and sum
    temp_storage_bytes_ = max(max_storage_bytes, sum_storage_bytes);
    d_temp_storage_buffer_.realloc(temp_storage_bytes_);
};

template <typename RealType> LogSumExp<RealType>::~LogSumExp(){};

template <typename RealType>
void LogSumExp<RealType>::sum_device(
    const int N,
    const RealType *d_values,
    RealType *d_exp_sum_out, // [2] First value is the max value, the second value is the log sum
    cudaStream_t stream) {
    if (N > N_) {
        throw std::runtime_error(
            "LogSumExp<RealType>::sum_device(): expected N < N_, got N=" + std::to_string(N) +
            ", N_=" + std::to_string(N_));
    }
    // Compute the max value, to subtract for the log computation
    gpuErrchk(cub::DeviceReduce::Max(d_temp_storage_buffer_.data, temp_storage_bytes_, d_values, d_exp_sum_out, N));

    int tpb = DEFAULT_THREADS_PER_BLOCK;
    // TBD: Combine Exp and Sum into a single kernel like energy accumulation
    k_exp_sub_max<<<ceil_divide(N, tpb), tpb, 0, stream>>>(N, d_exp_sum_out, d_values, d_temp_buffer_.data);
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cub::DeviceReduce::Sum(
        d_temp_storage_buffer_.data, temp_storage_bytes_, d_temp_buffer_.data, d_exp_sum_out + 1, N));
    // Skips the final `log` and addition of the max value call on the device, as it would be a kernel launch
}

template <typename RealType>
void LogSumExp<RealType>::sum_host(const int N, const RealType *h_values, RealType *h_out) {
    DeviceBuffer<RealType> d_vals(N);
    d_vals.copy_from(h_values);

    DeviceBuffer<RealType> d_out(2);

    cudaStream_t stream = static_cast<cudaStream_t>(0);
    this->sum_device(N, d_vals.data, d_out.data, stream);
    gpuErrchk(cudaStreamSynchronize(stream));

    RealType h_intermediate[2];
    d_out.copy_to(h_intermediate);
    // Compute the log and add back in the max value
    h_out[0] = compute_logsumexp_final<RealType>(
        h_intermediate[0],
        h_intermediate[1]); // Compute the log in host space, to be done by the consuming kernel later.
};

template class LogSumExp<double>;
template class LogSumExp<float>;

} // namespace timemachine
