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
      d_gumbel_(max_vals_per_segment_ * num_segments_), d_arg_max_(num_segments_), d_argmax_storage_(0) {

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
    const int *d_segment_offsets,        // [num_segments + 1]
    const RealType *d_log_probabilities, // [total_values]
    int *d_samples,                      // [num_segments]
    cudaStream_t stream) {
    // Perform sampling using the gumbel-max-trick like cupy
    // https://github.com/cupy/cupy/blob/46f6ed6e78661fb1d31b41a072519f119f5a2385/cupy/random/_generator.py#L1115-L1121
    curandErrchk(curandSetStream(cr_rng_, stream));
    curandErrchk(templateCurandUniform(cr_rng_, d_gumbel_.data, d_gumbel_.length));

    // Use the noise both as the noise and the buffer for the gumbel noise, does change the values in d_gumbel
    this->sample_given_noise_device(
        total_values,
        num_segments,
        d_segment_offsets,
        d_log_probabilities,
        d_gumbel_.data,
        d_gumbel_.data,
        d_samples,
        stream);
};

template <typename RealType>
void SegmentedWeightedRandomSampler<RealType>::sample_given_noise_device(
    const int total_values,
    const int num_segments,
    const int *d_segment_offsets,        // [num_segments + 1]
    const RealType *d_log_probabilities, // [total_values]
    const RealType *d_noise,             // [total_values]
    RealType *d_gumbel_noise,            // [total_values] Buffer to store the gumbel distribution
    int *d_samples,                      // [num_segments]
    cudaStream_t stream) {

    this->sample_given_noise_and_offset_device(
        total_values,
        num_segments,
        0, // Max offset can be safely ignored
        d_segment_offsets,
        d_log_probabilities,
        nullptr, // No noise offset
        d_noise,
        d_gumbel_noise,
        d_samples,
        stream);
}

template <typename RealType>
void SegmentedWeightedRandomSampler<RealType>::sample_given_gumbel_noise_device(
    const int num_segments,
    const int *d_segment_offsets,   // [num_segments + 1]
    const RealType *d_gumbel_noise, // [total_values]
    int *d_samples,
    cudaStream_t stream) {
    if (num_segments != num_segments_) {
        throw std::runtime_error(
            "SegmentedWeightedRandomerSampler::number of segments don't match: num_segments=" +
            std::to_string(num_segments) + ", num_segments_=" + std::to_string(num_segments_));
    }
    const int tpb = DEFAULT_THREADS_PER_BLOCK;
    gpuErrchk(cub::DeviceSegmentedReduce::ArgMax(
        d_argmax_storage_.data,
        temp_storage_bytes_,
        d_gumbel_noise,
        d_arg_max_.data,
        num_segments,
        d_segment_offsets,
        d_segment_offsets + 1,
        stream));

    k_copy_kv_key<RealType>
        <<<ceil_divide(num_segments, tpb), tpb, 0, stream>>>(num_segments, d_arg_max_.data, d_samples);
    gpuErrchk(cudaPeekAtLastError());
}

template <typename RealType>
void SegmentedWeightedRandomSampler<RealType>::sample_given_noise_and_offset_device(
    const int total_values,
    const int num_segments,
    const int max_offset,
    const int *d_segment_offsets,        // [num_segments + 1]
    const RealType *d_log_probabilities, // [total_values]
    const int *d_noise_offset,           // [total_values]
    const RealType *d_noise,             // [total_values]
    RealType *d_gumbel_noise,            // [total_values] Buffer to store the gumbel distribution
    int *d_samples,                      // [num_segments]
    cudaStream_t stream) {
    if (total_values > max_vals_per_segment_ * num_segments_) {
        throw std::runtime_error(
            "SegmentedWeightedRandomerSampler::total values is greater than buffer size:  vals_per_segment * "
            "num_segments=" +
            std::to_string(total_values) + ", buffer_size=" + std::to_string(max_vals_per_segment_ * num_segments_));
    }
    if (num_segments != num_segments_) {
        throw std::runtime_error(
            "SegmentedWeightedRandomerSampler::number of segments don't match: num_segments=" +
            std::to_string(num_segments) + ", num_segments_=" + std::to_string(num_segments_));
    }
    if (d_noise_offset != nullptr && max_offset <= 0) {
        throw std::runtime_error(
            "SegmentedWeightedRandomerSampler::when providing a noise offset, max offset must be greater than 0");
    }
    const int tpb = DEFAULT_THREADS_PER_BLOCK;

    const int blocks = ceil_divide(total_values, tpb);
    if (d_noise_offset == nullptr) {
        k_setup_gumbel_max_trick<<<blocks, tpb, 0, stream>>>(
            total_values, d_log_probabilities, d_noise, d_gumbel_noise);
        gpuErrchk(cudaPeekAtLastError());
    } else {
        dim3 dimGrid(blocks, num_segments, 1);
        k_setup_gumbel_max_trick_with_offset<<<dimGrid, tpb, 0, stream>>>(
            num_segments,
            total_values,
            max_offset,
            d_noise_offset,
            d_segment_offsets,
            d_log_probabilities,
            d_noise,
            d_gumbel_noise);
        gpuErrchk(cudaPeekAtLastError());
    }

    this->sample_given_gumbel_noise_device(num_segments, d_segment_offsets, d_gumbel_noise, d_samples, stream);
}

template <typename RealType>
std::vector<int>
SegmentedWeightedRandomSampler<RealType>::sample_host(const std::vector<std::vector<RealType>> &weights) {
    const int num_segments = static_cast<int>(weights.size());
    std::vector<int> h_selection(num_segments);
    std::vector<int> h_segments(num_segments + 1);

    const RealType inf = std::numeric_limits<RealType>::infinity();
    int offset = 0;
    h_segments[0] = offset;
    std::vector<RealType> h_log_probs;
    for (unsigned long i = 0; i < num_segments; i++) {
        const int num_vals = weights[i].size();
        if (num_vals == 0) {
            throw std::runtime_error("empty probability distribution not allowed");
        }
        offset += num_vals;
        h_segments[i + 1] = offset;
        // Convert the weights into log weights
        for (unsigned long j = 0; j < num_vals; j++) {
            RealType weight = weights[i][j];
            if (weight == inf) {
                throw std::runtime_error("unable to use infinity as a weight");
            } else if (isnan(weight)) {
                throw std::runtime_error("unable to use nan as a weight");
            } else if (weight < static_cast<RealType>(0.0)) {
                throw std::runtime_error("unable to use negative values as a weight");
            }

            RealType log_weight = log(weight);
            h_log_probs.push_back(log_weight);
        }
    }

    DeviceBuffer<RealType> d_log_probs(h_log_probs);
    DeviceBuffer<int> d_samples_buffer(num_segments);
    DeviceBuffer<int> d_segment_offsets(h_segments);

    cudaStream_t stream = static_cast<cudaStream_t>(0);
    this->sample_device(
        h_segments[num_segments],
        num_segments,
        d_segment_offsets.data,
        d_log_probs.data,
        d_samples_buffer.data,
        stream);
    gpuErrchk(cudaStreamSynchronize(stream));

    d_samples_buffer.copy_to(&h_selection[0]);
    return h_selection;
};

template class SegmentedWeightedRandomSampler<float>;
template class SegmentedWeightedRandomSampler<double>;

} // namespace timemachine
