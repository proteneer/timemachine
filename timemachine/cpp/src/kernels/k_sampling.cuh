#include <cub/cub.cuh>

namespace timemachine {

// References:
// https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
// https://lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/
template <typename RealType>
void __global__ k_setup_gumbel_max_trick(
    const int N,
    const RealType *__restrict__ log_weights,
    const RealType *__restrict__ gumbel_noise,
    RealType *__restrict__ prepared_gumbel);

template <typename T>
void __global__
k_copy_kv_key(const int N, const cub::KeyValuePair<int, T> *__restrict__ kv_pairs, int *__restrict__ out);

} // namespace timemachine
