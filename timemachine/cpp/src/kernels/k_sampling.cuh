#include <cub/cub.cuh>

namespace timemachine {

// References:
// https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
// https://lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/
template <typename RealType>
void __global__
k_setup_gumbel_max_trick(const int N, const RealType *__restrict__ log_weights, RealType *__restrict__ gumbel) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) {
        return;
    }

    const RealType gumbel_rand = -log(-log(gumbel[idx]));

    gumbel[idx] = log_weights[idx] + gumbel_rand;
}

template <typename T>
void __global__
k_copy_kv_key(const int N, const cub::KeyValuePair<int, T> *__restrict__ kv_pairs, int *__restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) {
        return;
    }

    // printf("Key %d Val %f\n", kv_pairs[idx].key, kv_pairs[idx].value);
    out[idx] = kv_pairs[idx].key;
}

} // namespace timemachine
