#include "k_logsumexp.cuh"

namespace timemachine {

template <typename RealType>
void __global__ k_compute_log_weights_from_energies(
    const int N, const RealType beta, const __int128 *__restrict__ energies, RealType *log_probabilities) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < N) {
        // TBD: What to with MAX_LLONG values?... Inf? But then extremely favorable energies will be treated as 'unfavorable'
        log_probabilities[idx] = beta * FIXED_ENERGY_TO_FLOAT<RealType>(energies[idx]);
        idx += gridDim.x * blockDim.x;
    }
}

// TBD - Needed?
template <typename RealType>
void __global__ k_compute_log_probs(
    const int N,
    const RealType *__restrict__ log_probabilities, // [N]
    const RealType *__restrict__ log_sum_exp,       // [2], first is sum of exp, second is max value
    RealType *__restrict__ normalized_log_probs     // [N]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // TBD Shared memory for the final value might be faster
    const RealType log_mean = compute_logsumexp_final<RealType>(log_sum_exp);
    while (idx < N) {
        normalized_log_probs[idx] = log_probabilities[idx] - log_mean;
        idx += gridDim.x * blockDim.x;
    }
}

} // namespace timemachine
