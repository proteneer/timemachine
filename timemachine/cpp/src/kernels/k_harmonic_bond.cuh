#include "../fixed_point.hpp"
#include "k_fixed_point.cuh"

namespace timemachine {

template <typename RealType>
void __global__ k_harmonic_bond(
    const int B, // number of bonds
    const double *__restrict__ coords,
    const double *__restrict__ params, // [B, 2]
    const int *__restrict__ bond_idxs, // [B, 2]
    unsigned long long *__restrict__ du_dx,
    unsigned long long *__restrict__ du_dp,
    __int128 *__restrict__ u) {

    int b_idx = blockDim.x * blockIdx.x + threadIdx.x;

    while (b_idx < B) {
        int src_idx = bond_idxs[b_idx * 2 + 0];
        int dst_idx = bond_idxs[b_idx * 2 + 1];

        RealType dx[3];
        RealType d2ij = 0;
#pragma unroll
        for (int d = 0; d < 3; d++) {
            RealType delta = coords[src_idx * 3 + d] - coords[dst_idx * 3 + d];
            dx[d] = delta;
            d2ij += delta * delta;
        }

        int kb_idx = b_idx * 2 + 0;
        int b0_idx = b_idx * 2 + 1;

        RealType kb = params[kb_idx];
        RealType b0 = params[b0_idx];

        RealType dij = sqrt(d2ij);
        RealType db = dij - b0;

        if (du_dx) {
            const RealType inv_dij = 1 / dij;
#pragma unroll
            for (int d = 0; d < 3; d++) {
                RealType grad_delta = b0 != 0 ? kb * db * dx[d] * inv_dij : kb * dx[d];
                atomicAdd(du_dx + src_idx * 3 + d, FLOAT_TO_FIXED_BONDED<RealType>(grad_delta));
                atomicAdd(du_dx + dst_idx * 3 + d, FLOAT_TO_FIXED_BONDED<RealType>(-grad_delta));
            }
        }

        if (du_dp) {
            atomicAdd(du_dp + kb_idx, FLOAT_TO_FIXED_BONDED(0.5 * db * db));
            atomicAdd(du_dp + b0_idx, FLOAT_TO_FIXED_BONDED(-kb * db));
        }

        if (u) {
            u[b_idx] = FLOAT_TO_FIXED_ENERGY<RealType>(kb / 2 * db * db);
        }
        b_idx += gridDim.x * blockDim.x;
    }
}

} // namespace timemachine
