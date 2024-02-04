#include "../fixed_point.hpp"
#include "../gpu_utils.cuh"
#include "k_fixed_point.cuh"

namespace timemachine {

template <typename RealType> inline __device__ RealType dot_product(const RealType a[3], const RealType b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename RealType>
inline __device__ void cross_product(const RealType a[3], const RealType b[3], RealType c[3]) {
    // these extra __dmul_rn calls are needed to preserve bitwise anticommutativity
    // i.e. cross(a,b) is bitwise identical to -cross(b,a) except in the sign-bit
    c[0] = rmul_rn(a[1], b[2]) - rmul_rn(a[2], b[1]);
    c[1] = rmul_rn(a[2], b[0]) - rmul_rn(a[0], b[2]);
    c[2] = rmul_rn(a[0], b[1]) - rmul_rn(a[1], b[0]);
}

template <typename RealType, int D>
void __global__ k_periodic_torsion(
    const int T,                          // number of bonds
    const double *__restrict__ coords,    // [n, 3]
    const double *__restrict__ params,    // [p, 3]
    const int *__restrict__ torsion_idxs, // [b, 4]
    unsigned long long *__restrict__ du_dx,
    unsigned long long *__restrict__ du_dp,
    __int128 *__restrict__ u) {

    const auto t_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (t_idx >= T) {
        return;
    }

    int i_idx = torsion_idxs[t_idx * 4 + 0];
    int j_idx = torsion_idxs[t_idx * 4 + 1];
    int k_idx = torsion_idxs[t_idx * 4 + 2];
    int l_idx = torsion_idxs[t_idx * 4 + 3];

    RealType r0[D];
    RealType r1[D];
    RealType r2[D];

    RealType r1_norm_square = 0;

    // (todo) cap to three dims, while keeping stride at 4
    for (int d = 0; d < D; d++) {
        r0[d] = coords[i_idx * D + d] - coords[j_idx * D + d];
        r1[d] = coords[k_idx * D + d] - coords[j_idx * D + d];
        r2[d] = coords[k_idx * D + d] - coords[l_idx * D + d];
        r1_norm_square += r1[d] * r1[d];
    }

    RealType r1_norm = sqrt(r1_norm_square);
    RealType n1[D], n2[D];

    cross_product(r0, r1, n1);
    cross_product(r1, r2, n2);

    RealType n1_norm_square, n2_norm_square;

    n1_norm_square = dot_product(n1, n1);
    n2_norm_square = dot_product(n2, n2);

    RealType n3[D];
    cross_product(n1, n2, n3);

    RealType d_angle_dR0[D];
    RealType d_angle_dR3[D];
    RealType d_angle_dR1[D];
    RealType d_angle_dR2[D];

    RealType r0_dot_r1 = dot_product(r0, r1);
    RealType r2_dot_r1 = dot_product(r2, r1);

    for (int d = 0; d < D; d++) {
        d_angle_dR0[d] = r1_norm / n1_norm_square * n1[d];
        d_angle_dR3[d] = -r1_norm / n2_norm_square * n2[d];
        d_angle_dR1[d] =
            (r0_dot_r1 / r1_norm_square - 1) * d_angle_dR0[d] - d_angle_dR3[d] * r2_dot_r1 / r1_norm_square;
        d_angle_dR2[d] =
            (r2_dot_r1 / r1_norm_square - 1) * d_angle_dR3[d] - d_angle_dR0[d] * r0_dot_r1 / r1_norm_square;
    }

    RealType r1_n = sqrt(dot_product(r1, r1));

    for (int d = 0; d < D; d++) {
        r1[d] /= r1_n;
    }

    RealType y = dot_product(n3, r1);
    RealType x = dot_product(n1, n2);
    RealType angle = atan2(y, x);

    int kt_idx = t_idx * D + 0;
    int phase_idx = t_idx * D + 1;
    int period_idx = t_idx * D + 2;

    RealType kt = params[kt_idx];
    RealType phase = params[phase_idx];
    RealType period = params[period_idx];

    RealType prefactor = -kt * sin(period * angle - phase) * period;

    if (du_dx) {
        for (int d = 0; d < D; d++) {
            atomicAdd(du_dx + i_idx * D + d, FLOAT_TO_FIXED_BONDED<RealType>(d_angle_dR0[d] * prefactor));
            atomicAdd(du_dx + j_idx * D + d, FLOAT_TO_FIXED_BONDED<RealType>(d_angle_dR1[d] * prefactor));
            atomicAdd(du_dx + k_idx * D + d, FLOAT_TO_FIXED_BONDED<RealType>(d_angle_dR2[d] * prefactor));
            atomicAdd(du_dx + l_idx * D + d, FLOAT_TO_FIXED_BONDED<RealType>(d_angle_dR3[d] * prefactor));
        }
    }

    if (du_dp) {
        RealType du_dkt = 1 + cos(period * angle - phase);
        RealType du_dphase = kt * sin(period * angle - phase);
        RealType du_dperiod = -kt * sin(period * angle - phase) * angle;

        atomicAdd(du_dp + kt_idx, FLOAT_TO_FIXED_BONDED(du_dkt));
        atomicAdd(du_dp + phase_idx, FLOAT_TO_FIXED_BONDED(du_dphase));
        atomicAdd(du_dp + period_idx, FLOAT_TO_FIXED_BONDED(du_dperiod));
    }

    if (u) {
        u[t_idx] = FLOAT_TO_FIXED_ENERGY<RealType>(kt * (1 + cos(period * angle - phase)));
    }
}

} // namespace timemachine
