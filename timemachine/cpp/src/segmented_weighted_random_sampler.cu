#include "device_buffer.hpp"
#include "gpu_utils.cuh"
#include "kernels/k_sampling.cuh"
#include "kernels/kernel_utils.cuh"
#include "math_utils.cuh"
#include "segmented_weighted_random_sampler.hpp"

namespace timemachine {

template <typename RealType>
SegmentedWeightedRandomSampler<RealType>::SegmentedWeightedRandomSampler(
    const int max_vals_per_segment, const int num_segments, const int seed)
    : max_vals_per_segment_(max_vals_per_segment), num_segments_(num_segments), temp_storage_bytes_(0),
      d_gumbel_(round_up_even(max_vals_per_segment_ * num_segments_)), d_arg_max_(num_segments_), d_argmax_storage_(0) {

    int *dummy_segments = nullptr;
    gpuErrchk(cub::DeviceSegmentedReduce::ArgMax(
        d_argmax_storage_.data,
        temp_storage_bytes_,
        d_gumbel_.data,
        d_arg_max_.data,
        num_segments_,
        dummy_segments,
        dummy_segments + 1));
    d_argmax_storage_.realloc(temp_storage_bytes_);

    curandErrchk(curandCreateGenerator(&cr_rng_, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrchk(curandSetPseudoRandomGeneratorSeed(cr_rng_, seed));
};

template <typename RealType> SegmentedWeightedRandomSampler<RealType>::~SegmentedWeightedRandomSampler() {
    curandErrchk(curandDestroyGenerator(cr_rng_));
};

template <typename RealType>
void SegmentedWeightedRandomSampler<RealType>::sample_device(
    const int total_values,
    const int num_segments,
    const int *d_segments,               // [num_segments + 1]
    const RealType *d_log_probabilities, // [total_values]
    int *d_samples,                      // [num_segments]
    cudaStream_t stream) {
    // Perform sampling using the gumbel-max-trick like cupy
    // https://github.com/cupy/cupy/blob/46f6ed6e78661fb1d31b41a072519f119f5a2385/cupy/random/_generator.py#L1115-L1121
    curandErrchk(curandSetStream(cr_rng_, stream));
    curandErrchk(templateCurandUniform(cr_rng_, d_gumbel_.data, d_gumbel_.length));

    // Use the noise both as the noise and the intermediate, does change the values in d_gumbel
    this->sample_given_noise_device(
        total_values, num_segments, d_segments, d_log_probabilities, d_gumbel_.data, d_gumbel_.data, d_samples, stream);
};

template <typename RealType>
void SegmentedWeightedRandomSampler<RealType>::sample_given_noise_device(
    const int total_values,
    const int num_segments,
    const int *d_segments,               // [num_segments]
    const RealType *d_log_probabilities, // [total_values]
    const RealType *d_noise,             // [total_values]
    RealType *d_intermediate,            // [total_values]
    int *d_samples,                      // [num_segments]
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
    const int blocks = ceil_divide(total_values, tpb);

    k_setup_gumbel_max_trick<<<blocks, tpb, 0, stream>>>(total_values, d_log_probabilities, d_noise, d_intermediate);
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cub::DeviceSegmentedReduce::ArgMax(
        d_argmax_storage_.data,
        temp_storage_bytes_,
        d_intermediate,
        d_arg_max_.data,
        num_segments_,
        d_segments,
        d_segments + 1));

    k_copy_kv_key<RealType>
        <<<ceil_divide(num_segments, tpb), tpb, 0, stream>>>(num_segments, d_arg_max_.data, d_samples);
    gpuErrchk(cudaPeekAtLastError());
}

template <typename RealType>
std::vector<int>
SegmentedWeightedRandomSampler<RealType>::sample_host(const std::vector<std::vector<RealType>> &probabilities) {
    const int num_segments = static_cast<int>(probabilities.size());
    std::vector<int> h_selection(num_segments);
    std::vector<int> h_segments(num_segments + 1);

    int offset = 0;
    h_segments[0] = offset;
    std::vector<RealType> h_log_probs;
    for (unsigned long i = 0; i < num_segments; i++) {
        offset += probabilities[i].size();
        h_segments[i + 1] = offset;
        // Convert the probabilities into log probabilities
        for (unsigned long j = 0; j < probabilities[i].size(); j++) {
            h_log_probs.push_back(log(probabilities[i][j]));
        }
    }

    DeviceBuffer<RealType> d_log_probs(h_log_probs);
    DeviceBuffer<int> d_samples_buffer(num_segments);
    DeviceBuffer<int> d_segments(h_segments);

    cudaStream_t stream = static_cast<cudaStream_t>(0);
    this->sample_device(
        h_segments[num_segments], num_segments, d_segments.data, d_log_probs.data, d_samples_buffer.data, stream);
    gpuErrchk(cudaStreamSynchronize(stream));

    d_samples_buffer.copy_to(&h_selection[0]);
    return h_selection;
};

template class SegmentedWeightedRandomSampler<float>;
template class SegmentedWeightedRandomSampler<double>;

} // namespace timemachine
