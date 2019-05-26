#pragma once

#include "surreal.cuh"

template<typename NumericType>
inline __device__ NumericType dot_product(
    NumericType x0, NumericType y0, NumericType z0,
    NumericType x1, NumericType y1, NumericType z1) {
    return x0*x1 + y0*y1 + z0*z1;
}

template<typename NumericType>
inline __device__ void cross_product(
    NumericType a1, NumericType a2, NumericType a3,
    NumericType b1, NumericType b2, NumericType b3,
    NumericType &s1, NumericType &s2, NumericType &s3) {

    s1 = a2*b3 - a3*b2;
    s2 = a3*b1 - a1*b3;
    s3 = a1*b2 - a2*b1;

}

inline __device__ float atan2(double x, float y) {
    return atan2(float(x), float(y));
}

template<typename RealType>
inline __device__ RealType norm(RealType x, RealType y, RealType z) {
    return sqrt(x*x + y*y + z*z);
}

template<typename RealType>
void __global__ k_periodic_torsion_derivatives(
    const int num_atoms,    // n
    const int num_params,   // p
    const RealType *coords, // [n, 3]
    const RealType *params, // [p,]
    const int num_angles,    // a
    const int *torsion_idxs,   // [a, 4]
    const int *param_idxs,  // [a, 3]
    RealType *E,            // [,] never null
    RealType *dE_dx,        // [n,3] or null
    // parameters used for computing derivatives
    const RealType *dx_dp, // [dp, n, 3]
    const int *dp_idxs,     // of shape [dp] or null, if null then we don't compute parameter derivatives
    RealType *dE_dp,        // [dp,] or null
    RealType *d2E_dxdp      // [dp, n, 3] or null
) {
    auto conf_idx = blockIdx.z;

    // note that dp == gridDim.y
    const bool compute_dp = (dp_idxs != nullptr);
    auto a_idx = blockDim.x*blockIdx.x + threadIdx.x;

    if(a_idx >= num_angles) {
        return;
    }

    int atom_0_idx = torsion_idxs[a_idx*4+0];
    int atom_1_idx = torsion_idxs[a_idx*4+1];
    int atom_2_idx = torsion_idxs[a_idx*4+2];
    int atom_3_idx = torsion_idxs[a_idx*4+3];

    RealType rx0 = coords[conf_idx*num_atoms*3+atom_0_idx*3+0];
    RealType ry0 = coords[conf_idx*num_atoms*3+atom_0_idx*3+1];
    RealType rz0 = coords[conf_idx*num_atoms*3+atom_0_idx*3+2];

    RealType rx1 = coords[conf_idx*num_atoms*3+atom_1_idx*3+0];
    RealType ry1 = coords[conf_idx*num_atoms*3+atom_1_idx*3+1];
    RealType rz1 = coords[conf_idx*num_atoms*3+atom_1_idx*3+2];

    RealType rx2 = coords[conf_idx*num_atoms*3+atom_2_idx*3+0];
    RealType ry2 = coords[conf_idx*num_atoms*3+atom_2_idx*3+1];
    RealType rz2 = coords[conf_idx*num_atoms*3+atom_2_idx*3+2];

    RealType rx3 = coords[conf_idx*num_atoms*3+atom_3_idx*3+0];
    RealType ry3 = coords[conf_idx*num_atoms*3+atom_3_idx*3+1];
    RealType rz3 = coords[conf_idx*num_atoms*3+atom_3_idx*3+2];

    RealType ix0 = 0;
    RealType iy0 = 0;
    RealType iz0 = 0;

    RealType ix1 = 0;
    RealType iy1 = 0;
    RealType iz1 = 0;

    RealType ix2 = 0;
    RealType iy2 = 0;
    RealType iz2 = 0;

    RealType ix3 = 0;
    RealType iy3 = 0;
    RealType iz3 = 0;

    RealType rk = params[param_idxs[a_idx*3+0]];
    RealType rphase = params[param_idxs[a_idx*3+1]];
    RealType rperiod = params[param_idxs[a_idx*3+2]];

    RealType ik = 0;
    RealType iphase = 0;
    RealType iperiod = 0;

    const auto dp_idx = blockIdx.y;
    const auto num_dp = gridDim.y;

    const RealType step_size = 1e-7;

    if(compute_dp) {

        // complex-step coordinates
        if(dx_dp != nullptr) {
            ix0 = dx_dp[conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+atom_0_idx*3+0]*step_size;
            iy0 = dx_dp[conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+atom_0_idx*3+1]*step_size;
            iz0 = dx_dp[conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+atom_0_idx*3+2]*step_size;

            ix1 = dx_dp[conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+atom_1_idx*3+0]*step_size;
            iy1 = dx_dp[conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+atom_1_idx*3+1]*step_size;
            iz1 = dx_dp[conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+atom_1_idx*3+2]*step_size;

            ix2 = dx_dp[conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+atom_2_idx*3+0]*step_size;
            iy2 = dx_dp[conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+atom_2_idx*3+1]*step_size;
            iz2 = dx_dp[conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+atom_2_idx*3+2]*step_size;

            ix3 = dx_dp[conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+atom_3_idx*3+0]*step_size;
            iy3 = dx_dp[conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+atom_3_idx*3+1]*step_size;
            iz3 = dx_dp[conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+atom_3_idx*3+2]*step_size;
        }

        // complex-step parameters 
        if(param_idxs[a_idx*3+0] == dp_idxs[dp_idx]) {
            ik = step_size;
        }
        if(param_idxs[a_idx*3+1] == dp_idxs[dp_idx]) {
            iphase = step_size;
        }
        if(param_idxs[a_idx*3+2] == dp_idxs[dp_idx]) {
            iperiod = step_size;
        }   
    }

    Surreal<RealType> x0(rx0, ix0);
    Surreal<RealType> y0(ry0, iy0);
    Surreal<RealType> z0(rz0, iz0);

    Surreal<RealType> x1(rx1, ix1);
    Surreal<RealType> y1(ry1, iy1);
    Surreal<RealType> z1(rz1, iz1);

    Surreal<RealType> x2(rx2, ix2);
    Surreal<RealType> y2(ry2, iy2);
    Surreal<RealType> z2(rz2, iz2);

    Surreal<RealType> x3(rx3, ix3);
    Surreal<RealType> y3(ry3, iy3);
    Surreal<RealType> z3(rz3, iz3);

    Surreal<RealType> k(rk, ik);
    Surreal<RealType> phase(rphase, iphase);
    Surreal<RealType> period(rperiod, iperiod);

    Surreal<RealType> rij_x = x1 - x0;
    Surreal<RealType> rij_y = y1 - y0;
    Surreal<RealType> rij_z = z1 - z0;

    Surreal<RealType> rkj_x = x1 - x2;
    Surreal<RealType> rkj_y = y1 - y2;
    Surreal<RealType> rkj_z = z1 - z2;

    Surreal<RealType> rkj_norm = norm(rkj_x, rkj_y, rkj_z);
    Surreal<RealType> rkj_norm_square = dot_product(rkj_x, rkj_y, rkj_z, rkj_x, rkj_y, rkj_z);

    Surreal<RealType> rkl_x = x3 - x2;
    Surreal<RealType> rkl_y = y3 - y2;
    Surreal<RealType> rkl_z = z3 - z2;

    Surreal<RealType> n1_x, n1_y, n1_z, n2_x, n2_y, n2_z;

    cross_product(rij_x, rij_y, rij_z, rkj_x, rkj_y, rkj_z, n1_x, n1_y, n1_z);
    cross_product(rkj_x, rkj_y, rkj_z, rkl_x, rkl_y, rkl_z, n2_x, n2_y, n2_z);

    Surreal<RealType> n1_norm_square = dot_product(n1_x, n1_y, n1_z, n1_x, n1_y, n1_z);
    Surreal<RealType> n2_norm_square = dot_product(n2_x, n2_y, n2_z, n2_x, n2_y, n2_z);

    Surreal<RealType> n3_x, n3_y, n3_z;

    cross_product(n1_x, n1_y, n1_z, n2_x, n2_y, n2_z, n3_x, n3_y, n3_z);

    Surreal<RealType> dangle_dR0_x = rkj_norm/(n1_norm_square) * n1_x;
    Surreal<RealType> dangle_dR0_y = rkj_norm/(n1_norm_square) * n1_y;
    Surreal<RealType> dangle_dR0_z = rkj_norm/(n1_norm_square) * n1_z; 

    Surreal<RealType> dangle_dR3_x = -rkj_norm/(n2_norm_square) * n2_x;
    Surreal<RealType> dangle_dR3_y = -rkj_norm/(n2_norm_square) * n2_y;
    Surreal<RealType> dangle_dR3_z = -rkj_norm/(n2_norm_square) * n2_z; 

    Surreal<RealType> dangle_dR1_x = (dot_product(rij_x, rij_y, rij_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square) - 1) * dangle_dR0_x - dangle_dR3_x*dot_product(rkl_x, rkl_y, rkl_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square);
    Surreal<RealType> dangle_dR1_y = (dot_product(rij_x, rij_y, rij_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square) - 1) * dangle_dR0_y - dangle_dR3_y*dot_product(rkl_x, rkl_y, rkl_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square);
    Surreal<RealType> dangle_dR1_z = (dot_product(rij_x, rij_y, rij_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square) - 1) * dangle_dR0_z - dangle_dR3_z*dot_product(rkl_x, rkl_y, rkl_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square);

    Surreal<RealType> dangle_dR2_x = (dot_product(rkl_x, rkl_y, rkl_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square) - 1) * dangle_dR3_x - dangle_dR0_x*dot_product(rij_x, rij_y, rij_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square);
    Surreal<RealType> dangle_dR2_y = (dot_product(rkl_x, rkl_y, rkl_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square) - 1) * dangle_dR3_y - dangle_dR0_y*dot_product(rij_x, rij_y, rij_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square);
    Surreal<RealType> dangle_dR2_z = (dot_product(rkl_x, rkl_y, rkl_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square) - 1) * dangle_dR3_z - dangle_dR0_z*dot_product(rij_x, rij_y, rij_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square);

    Surreal<RealType> rkj_n = norm(rkj_x, rkj_y, rkj_z);
    rkj_x /= rkj_n;
    rkj_y /= rkj_n;
    rkj_z /= rkj_n;

    Surreal<RealType> y = dot_product(n3_x, n3_y, n3_z, rkj_x, rkj_y, rkj_z);
    Surreal<RealType> x = dot_product(n1_x, n1_y, n1_z, n2_x, n2_y, n2_z);
    Surreal<RealType> angle = atan2(y, x); // should this

    Surreal<RealType> prefactor = k*sin(period*angle - phase)*period;

    Surreal<RealType> atom_0_grad_x = dangle_dR0_x * prefactor;
    Surreal<RealType> atom_0_grad_y = dangle_dR0_y * prefactor;
    Surreal<RealType> atom_0_grad_z = dangle_dR0_z * prefactor;

    Surreal<RealType> atom_1_grad_x = dangle_dR1_x * prefactor;
    Surreal<RealType> atom_1_grad_y = dangle_dR1_y * prefactor;
    Surreal<RealType> atom_1_grad_z = dangle_dR1_z * prefactor;

    Surreal<RealType> atom_2_grad_x = dangle_dR2_x * prefactor;
    Surreal<RealType> atom_2_grad_y = dangle_dR2_y * prefactor;
    Surreal<RealType> atom_2_grad_z = dangle_dR2_z * prefactor;

    Surreal<RealType> atom_3_grad_x = dangle_dR3_x * prefactor;
    Surreal<RealType> atom_3_grad_y = dangle_dR3_y * prefactor;
    Surreal<RealType> atom_3_grad_z = dangle_dR3_z * prefactor;

    Surreal<RealType> energy = k*(1+cos(period*angle - phase));

    // E is never null
    if(blockIdx.y == 0) {
        atomicAdd(E + conf_idx, energy.real);
    }

    if(blockIdx.y == 0 && dE_dx != nullptr) {
        atomicAdd(dE_dx+conf_idx*num_atoms*3+atom_0_idx*3+0, atom_0_grad_x.real);
        atomicAdd(dE_dx+conf_idx*num_atoms*3+atom_0_idx*3+1, atom_0_grad_y.real);
        atomicAdd(dE_dx+conf_idx*num_atoms*3+atom_0_idx*3+2, atom_0_grad_z.real);

        atomicAdd(dE_dx+conf_idx*num_atoms*3+atom_1_idx*3+0, atom_1_grad_x.real);
        atomicAdd(dE_dx+conf_idx*num_atoms*3+atom_1_idx*3+1, atom_1_grad_y.real);
        atomicAdd(dE_dx+conf_idx*num_atoms*3+atom_1_idx*3+2, atom_1_grad_z.real);

        atomicAdd(dE_dx+conf_idx*num_atoms*3+atom_2_idx*3+0, atom_2_grad_x.real);
        atomicAdd(dE_dx+conf_idx*num_atoms*3+atom_2_idx*3+1, atom_2_grad_y.real);
        atomicAdd(dE_dx+conf_idx*num_atoms*3+atom_2_idx*3+2, atom_2_grad_z.real);

        atomicAdd(dE_dx+conf_idx*num_atoms*3+atom_3_idx*3+0, atom_3_grad_x.real);
        atomicAdd(dE_dx+conf_idx*num_atoms*3+atom_3_idx*3+1, atom_3_grad_y.real);
        atomicAdd(dE_dx+conf_idx*num_atoms*3+atom_3_idx*3+2, atom_3_grad_z.real);
    }

    if(compute_dp && dE_dp != nullptr) {
        atomicAdd(dE_dp+conf_idx*num_dp+dp_idx, energy.imag/step_size);
    }

    if(compute_dp && d2E_dxdp != nullptr) {
        atomicAdd(d2E_dxdp+conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+atom_0_idx*3 + 0, atom_0_grad_x.imag/step_size);
        atomicAdd(d2E_dxdp+conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+atom_0_idx*3 + 1, atom_0_grad_y.imag/step_size);
        atomicAdd(d2E_dxdp+conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+atom_0_idx*3 + 2, atom_0_grad_z.imag/step_size);

        atomicAdd(d2E_dxdp+conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+atom_1_idx*3 + 0, atom_1_grad_x.imag/step_size);
        atomicAdd(d2E_dxdp+conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+atom_1_idx*3 + 1, atom_1_grad_y.imag/step_size);
        atomicAdd(d2E_dxdp+conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+atom_1_idx*3 + 2, atom_1_grad_z.imag/step_size);

        atomicAdd(d2E_dxdp+conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+atom_2_idx*3 + 0, atom_2_grad_x.imag/step_size);
        atomicAdd(d2E_dxdp+conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+atom_2_idx*3 + 1, atom_2_grad_y.imag/step_size);
        atomicAdd(d2E_dxdp+conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+atom_2_idx*3 + 2, atom_2_grad_z.imag/step_size);

        atomicAdd(d2E_dxdp+conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+atom_3_idx*3 + 0, atom_3_grad_x.imag/step_size);
        atomicAdd(d2E_dxdp+conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+atom_3_idx*3 + 1, atom_3_grad_y.imag/step_size);
        atomicAdd(d2E_dxdp+conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+atom_3_idx*3 + 2, atom_3_grad_z.imag/step_size);
    }
}

