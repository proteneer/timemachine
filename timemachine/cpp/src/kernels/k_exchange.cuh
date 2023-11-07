#include "k_logsumexp.cuh"

namespace timemachine {

// When we are considering exchange we want to treat Nan probabilities as inf
// Allows us to go from a clashy state to a non-clashy state. And no nan poisoning
template <typename RealType> RealType __host__ __device__ convert_nan_to_inf(const RealType input) {
    if (isnan(input)) {
        return INFINITY;
    }
    return input;
}

template <typename RealType>
void __global__ k_attempt_exchange_move(
    const int N,
    const RealType *__restrict__ rand,               // [1]
    const RealType *__restrict__ before_log_sum_exp, // [2]
    const RealType *__restrict__ after_log_sum_exp,  // [2]
    const double *__restrict__ moved_coords,         // [N, 3]
    double *__restrict__ dest_coords,                // [N, 3]
    size_t *__restrict__ num_accepted                // [1]
) {
    int atom_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // All kernels compute the same acceptance
    // TBD investigate shared memory for speed
    RealType before_log_prob = convert_nan_to_inf(compute_logsumexp_final<RealType>(before_log_sum_exp));
    RealType after_log_prob = convert_nan_to_inf(compute_logsumexp_final<RealType>(after_log_sum_exp));

    RealType log_acceptance_prob = min(before_log_prob - after_log_prob, static_cast<RealType>(0.0));
    const bool accepted = rand[0] < exp(log_acceptance_prob);
    if (atom_idx == 0 && accepted) {
        num_accepted[0]++;
    }

    // If accepted, move the coords into place
    while (accepted && atom_idx < N) {
        dest_coords[atom_idx * 3 + 0] = moved_coords[atom_idx * 3 + 0];
        dest_coords[atom_idx * 3 + 1] = moved_coords[atom_idx * 3 + 1];
        dest_coords[atom_idx * 3 + 2] = moved_coords[atom_idx * 3 + 2];
        atom_idx += gridDim.x * blockDim.x;
    }
}

template <typename RealType>
void __global__ k_store_accepted_log_probability(
    const int num_weights,
    const RealType *__restrict__ rand,              // [1]
    RealType *__restrict__ before_log_sum_exp,      // [2]
    const RealType *__restrict__ after_log_sum_exp, // [2]
    RealType *__restrict__ before_weights,          // [num_weights]
    const RealType *__restrict__ after_weights      // [num_weights]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0) {
        return; // Only one block can run this
    }

    RealType before_log_prob = convert_nan_to_inf(compute_logsumexp_final<RealType>(before_log_sum_exp));
    RealType after_log_prob = convert_nan_to_inf(compute_logsumexp_final<RealType>(after_log_sum_exp));

    RealType log_acceptance_prob = min(before_log_prob - after_log_prob, static_cast<RealType>(0.0));
    const bool accepted = rand[0] < exp(log_acceptance_prob);
    __syncthreads();
    if (!accepted) {
        return;
    }
    // Swap the values after all threads have computed the log probability
    if (idx == 0) {
        before_log_sum_exp[0] = after_log_sum_exp[0];
        before_log_sum_exp[1] = after_log_sum_exp[1];
    }
    // Copy over the weights
    while (idx < num_weights) {
        before_weights[idx] = after_weights[idx];
        idx += gridDim.x * blockDim.x;
    }
}

} // namespace timemachine
