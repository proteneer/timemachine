#include "k_logsumexp.cuh"

namespace timemachine {

// Convert the outputs of LogSumExp kernels into the final logsumexp value

template <typename RealType>
void __global__ k_exp_sub_max(
    const int N,
    const RealType *__restrict__ max,  // [1]
    const RealType *__restrict__ vals, // [N]
    RealType *__restrict__ out         // [N]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    out[idx] = exp(vals[idx] - max[0]);
}

template void __global__ k_exp_sub_max<float>(const int, const float *, const float *, float *);
template void __global__ k_exp_sub_max<double>(const int, const double *, const double *, double *);

} // namespace timemachine
