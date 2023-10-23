#include "device_buffer.hpp"
#include "gpu_utils.cuh"
#include "kernels/k_sampling.cuh"
#include "kernels/kernel_utils.cuh"
#include "math_utils.cuh"
#include "random_sampler.hpp"

namespace timemachine {

template <typename RealType>
RandomSampler<RealType>::RandomSampler(const int N, const int seed)
    : N_(N), temp_storage_bytes_(0), d_rand_(round_up_even(N_)), d_gumbel_(N_) {

    // Allocate enough space for a single key value pair
    cudaSafeMalloc(&d_arg_max_, 1 * sizeof(cub::KeyValuePair<int, RealType>));
    gpuErrchk(cub::DeviceReduce::ArgMax(
        nullptr, temp_storage_bytes_, d_gumbel_.data, static_cast<cub::KeyValuePair<int, RealType> *>(d_arg_max_), N_));
    d_sort_storage_.reset(new DeviceBuffer<char>(temp_storage_bytes_));

    curandErrchk(curandCreateGenerator(&cr_rng_, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrchk(curandSetPseudoRandomGeneratorSeed(cr_rng_, seed));
};

template <typename RealType> RandomSampler<RealType>::~RandomSampler() {
    curandErrchk(curandDestroyGenerator(cr_rng_));

    gpuErrchk(cudaFree(d_arg_max_));
};

template <typename RealType>
void RandomSampler<RealType>::sample_device(
    const int N, const int K, const RealType *d_log_probabilities, int *d_samples, cudaStream_t stream) {
    if (N > N_) {
        throw std::runtime_error(
            "N is greater than buffer size:  N=" + std::to_string(N) + ", N_=" + std::to_string(N_));
    }
    if (K <= 0) {
        throw std::runtime_error("K must be at least 1: K=" + std::to_string(K));
    }
    // Perform sampling using the gumbel-max-trick like cupy
    // https://github.com/cupy/cupy/blob/46f6ed6e78661fb1d31b41a072519f119f5a2385/cupy/random/_generator.py#L1115-L1121
    curandErrchk(curandSetStream(cr_rng_, stream));
    const int tpb = DEFAULT_THREADS_PER_BLOCK;
    const int blocks = ceil_divide(N, tpb);
    for (int i = 0; i < K; i++) {
        curandErrchk(templateCurandUniform(cr_rng_, d_rand_.data, round_up_even(N)));

        k_setup_gumbel_max_trick<<<blocks, tpb, 0, stream>>>(N, d_rand_.data, d_log_probabilities, d_gumbel_.data);
        gpuErrchk(cudaPeekAtLastError());

        gpuErrchk(cub::DeviceReduce::ArgMax(
            d_sort_storage_->data,
            temp_storage_bytes_,
            d_gumbel_.data,
            static_cast<cub::KeyValuePair<int, RealType> *>(d_arg_max_),
            N));

        k_copy_kv_key<RealType>
            <<<1, 1, 0, stream>>>(1, static_cast<cub::KeyValuePair<int, RealType> *>(d_arg_max_), d_samples + i);
        gpuErrchk(cudaPeekAtLastError());
    }
};

template <typename RealType>
std::vector<int> RandomSampler<RealType>::sample_host(const int K, const std::vector<RealType> &probabilities) {
    std::vector<int> h_selection(K);
    // Convert the probabilities into log probabilities
    std::vector<RealType> h_log_probs(probabilities.size());
    for (int i = 0; i < probabilities.size(); i++) {
        h_log_probs[i] = log(probabilities[i]);
    }

    DeviceBuffer<int> d_samples_buffer(K);
    DeviceBuffer<RealType> d_probability_buffer_(h_log_probs.size());
    d_probability_buffer_.copy_from(&h_log_probs[0]);

    cudaStream_t stream = static_cast<cudaStream_t>(0);
    this->sample_device(h_log_probs.size(), K, d_probability_buffer_.data, d_samples_buffer.data, stream);
    gpuErrchk(cudaStreamSynchronize(stream));

    d_samples_buffer.copy_to(&h_selection[0]);
    return h_selection;
};

template class RandomSampler<float>;
template class RandomSampler<double>;

} // namespace timemachine
