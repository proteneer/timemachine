#include "../fixed_point.hpp"
#include "k_fixed_point.cuh"

template <typename RealType, int D>
void __global__ k_harmonic_angle_inference(
    const int A,                       // number of bonds
    const double *__restrict__ coords, // [N, 3]
    const double *__restrict__ params, // [P, 2]
    const double lambda,
    const int *__restrict__ lambda_mult,
    const int *__restrict__ lambda_offset,
    const int *__restrict__ angle_idxs, // [A, 3]
    unsigned long long *__restrict__ du_dx,
    unsigned long long *__restrict__ du_dp,
    unsigned long long *__restrict__ du_dl,
    unsigned long long *__restrict__ u) {

    const auto a_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (a_idx >= A) {
        return;
    }

    int i_idx = angle_idxs[a_idx * 3 + 0];
    int j_idx = angle_idxs[a_idx * 3 + 1];
    int k_idx = angle_idxs[a_idx * 3 + 2];

    RealType rij[3];
    RealType rjk[3];
    RealType nij = 0; // initialize your summed variables!
    RealType njk = 0; // initialize your summed variables!
    RealType top = 0;
    // this is a little confusing
    for (int d = 0; d < 3; d++) {
        RealType vij = coords[j_idx * D + d] - coords[i_idx * D + d];
        RealType vjk = coords[j_idx * D + d] - coords[k_idx * D + d];

        rij[d] = vij;
        rjk[d] = vjk;
        nij += vij * vij;
        njk += vjk * vjk;

        top += vij * vjk;
    }

    nij = sqrt(nij);
    njk = sqrt(njk);

    RealType nijk = nij * njk;
    RealType n3ij = nij * nij * nij;
    RealType n3jk = njk * njk * njk;

    int ka_idx = a_idx * 2 + 0;
    int a0_idx = a_idx * 2 + 1;

    RealType ka = params[ka_idx];
    RealType a0 = params[a0_idx];

    RealType delta = top / nijk - cos(a0);

    RealType prefactor = lambda_offset[a_idx] + lambda_mult[a_idx] * lambda;

    if (du_dx) {
        for (int d = 0; d < 3; d++) {
            RealType grad_i = ka * delta * (rij[d] * top / (n3ij * njk) + (-rjk[d]) / nijk);
            atomicAdd(du_dx + i_idx * D + d, FLOAT_TO_FIXED_BONDED<RealType>(grad_i * prefactor));

            RealType grad_j =
                ka * delta *
                ((-rij[d] * top / (n3ij * njk) + (-rjk[d]) * top / (nij * n3jk) + (rij[d] + rjk[d]) / nijk));
            atomicAdd(du_dx + j_idx * D + d, FLOAT_TO_FIXED_BONDED<RealType>(grad_j * prefactor));

            RealType grad_k = ka * delta * (-rij[d] / nijk + rjk[d] * top / (nij * n3jk));
            atomicAdd(du_dx + k_idx * D + d, FLOAT_TO_FIXED_BONDED<RealType>(grad_k * prefactor));
        }
    }

    if (du_dp) {
        RealType dka_grad = delta * delta / 2;
        atomicAdd(du_dp + ka_idx, FLOAT_TO_FIXED_BONDED(dka_grad * prefactor));
        RealType da0_grad = delta * ka * sin(a0);
        atomicAdd(du_dp + a0_idx, FLOAT_TO_FIXED_BONDED(da0_grad * prefactor));
    }

    if (u) {
        atomicAdd(u + i_idx, FLOAT_TO_FIXED_BONDED(ka / 2 * delta * delta * prefactor));
    }

    if (du_dl) {
        atomicAdd(du_dl + i_idx, FLOAT_TO_FIXED_BONDED(lambda_mult[a_idx] * ka / 2 * delta * delta));
    }
}
