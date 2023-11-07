#pragma once

namespace timemachine {

// Convert the outputs of LogSumExp kernels into the final logsumexp value
template <typename RealType>
RealType __host__ __device__ __forceinline__ compute_logsumexp_final(const RealType *__restrict__ log_sum_exp_res // [2]
) {
    return log(log_sum_exp_res[1]) + log_sum_exp_res[0];
}

template <typename RealType>
void __global__ k_exp_sub_max(
    const int N,
    const RealType *__restrict__ max,  // [1]
    const RealType *__restrict__ vals, // [N]
    RealType *__restrict__ out         // [N]
);

} // namespace timemachine
