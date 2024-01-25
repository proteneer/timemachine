#include "k_sampling.cuh"
#include <assert.h>

namespace timemachine {

// References:
// https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
// https://lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/
template <typename RealType>
void __global__ k_setup_gumbel_max_trick(
    const int N,
    const RealType *__restrict__ log_weights,
    const RealType *__restrict__ gumbel_noise,
    RealType *__restrict__ prepared_gumbel) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) {
        return;
    }

    const RealType weight = log_weights[idx];
    assert(!isnan(weight));

    const RealType gumbel_rand = -log(-log(gumbel_noise[idx]));

    prepared_gumbel[idx] = weight + gumbel_rand;
}

template void __global__ k_setup_gumbel_max_trick<float>(
    const int N,
    const float *__restrict__ log_weights,
    const float *__restrict__ gumbel_noise,
    float *__restrict__ prepared_gumbel);

template void __global__ k_setup_gumbel_max_trick<double>(
    const int N,
    const double *__restrict__ log_weights,
    const double *__restrict__ gumbel_noise,
    double *__restrict__ prepared_gumbel);

template <typename T>
void __global__
k_copy_kv_key(const int N, const cub::KeyValuePair<int, T> *__restrict__ kv_pairs, int *__restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) {
        return;
    }

    out[idx] = kv_pairs[idx].key;
}

template void __global__
k_copy_kv_key<float>(const int N, const cub::KeyValuePair<int, float> *__restrict__ kv_pairs, int *__restrict__ out);

template void __global__
k_copy_kv_key<double>(const int N, const cub::KeyValuePair<int, double> *__restrict__ kv_pairs, int *__restrict__ out);

} // namespace timemachine
