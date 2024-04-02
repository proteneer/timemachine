#include "k_logsumexp.cuh"
#include <cmath>

namespace timemachine {

template <typename RealType>
void __global__ k_compute_log_weights_from_energies(
    const int N, const RealType beta, const __int128 *__restrict__ energies, RealType *log_probabilities) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __int128 energy;
    while (idx < N) {
        energy = energies[idx];
        // If the energy is invalid, set the log probability to inf
        log_probabilities[idx] = !fixed_point_overflow(energy)
                                     ? FIXED_ENERGY_TO_FLOAT<RealType>(FLOAT_TO_FIXED_ENERGY<RealType>(
                                           beta * FIXED_ENERGY_TO_FLOAT<RealType>(energy)))
                                     : INFINITY;
        idx += gridDim.x * blockDim.x;
    }
}

} // namespace timemachine
