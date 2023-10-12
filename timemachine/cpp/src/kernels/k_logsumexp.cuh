
namespace timemachine {

template <typename RealType>
void __global__ k_exp_sub_max(
    const int N,
    const RealType *__restrict__ max, // [0]
    RealType *__restrict__ vals       // [N]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    vals[idx] = exp(vals[idx] - max[0]);
}

} // namespace timemachine
