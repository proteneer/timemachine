#pragma once

#include "surreal.cuh"

template<typename RealType>
inline __device__ RealType dot_product(
    RealType x0, RealType y0, RealType z0,
    RealType x1, RealType y1, RealType z1) {
    return x0*x1 + y0*y1 + z0*z1;
}

template<typename RealType>
inline __device__ void cross_product(
    RealType a1, RealType a2, RealType a3,
    RealType b1, RealType b2, RealType b3,
    RealType &s1, RealType &s2, RealType &s3) {

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

template <typename CoordType, typename ParamType, typename OutType, int NDIMS> 
inline __device__ OutType torsion_gradient(
    const CoordType *xs,
    const ParamType *params,
    OutType *grads) {

    CoordType x0 = xs[0];
    CoordType y0 = xs[1];
    CoordType z0 = xs[2];

    CoordType x1 = xs[3];
    CoordType y1 = xs[4];
    CoordType z1 = xs[5];

    CoordType x2 = xs[6];
    CoordType y2 = xs[7];
    CoordType z2 = xs[8];

    CoordType x3 = xs[9];
    CoordType y3 = xs[10];
    CoordType z3 = xs[11];

    ParamType k = params[0];
    ParamType phase = params[1];
    ParamType period = params[2];

    CoordType rij_x = x1 - x0;
    CoordType rij_y = y1 - y0;
    CoordType rij_z = z1 - z0;

    CoordType rkj_x = x1 - x2;
    CoordType rkj_y = y1 - y2;
    CoordType rkj_z = z1 - z2;

    CoordType rkj_norm = norm(rkj_x, rkj_y, rkj_z);
    CoordType rkj_norm_square = dot_product(rkj_x, rkj_y, rkj_z, rkj_x, rkj_y, rkj_z);

    CoordType rkl_x = x3 - x2;
    CoordType rkl_y = y3 - y2;
    CoordType rkl_z = z3 - z2;

    CoordType n1_x, n1_y, n1_z, n2_x, n2_y, n2_z;

    cross_product(rij_x, rij_y, rij_z, rkj_x, rkj_y, rkj_z, n1_x, n1_y, n1_z);
    cross_product(rkj_x, rkj_y, rkj_z, rkl_x, rkl_y, rkl_z, n2_x, n2_y, n2_z);

    CoordType n1_norm_square = dot_product(n1_x, n1_y, n1_z, n1_x, n1_y, n1_z);
    CoordType n2_norm_square = dot_product(n2_x, n2_y, n2_z, n2_x, n2_y, n2_z);

    CoordType n3_x, n3_y, n3_z;

    cross_product(n1_x, n1_y, n1_z, n2_x, n2_y, n2_z, n3_x, n3_y, n3_z);

    CoordType dangle_dR0_x = rkj_norm/(n1_norm_square) * n1_x;
    CoordType dangle_dR0_y = rkj_norm/(n1_norm_square) * n1_y;
    CoordType dangle_dR0_z = rkj_norm/(n1_norm_square) * n1_z; 

    CoordType dangle_dR3_x = -rkj_norm/(n2_norm_square) * n2_x;
    CoordType dangle_dR3_y = -rkj_norm/(n2_norm_square) * n2_y;
    CoordType dangle_dR3_z = -rkj_norm/(n2_norm_square) * n2_z; 

    CoordType dangle_dR1_x = (dot_product(rij_x, rij_y, rij_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square) - 1) * dangle_dR0_x - dangle_dR3_x*dot_product(rkl_x, rkl_y, rkl_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square);
    CoordType dangle_dR1_y = (dot_product(rij_x, rij_y, rij_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square) - 1) * dangle_dR0_y - dangle_dR3_y*dot_product(rkl_x, rkl_y, rkl_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square);
    CoordType dangle_dR1_z = (dot_product(rij_x, rij_y, rij_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square) - 1) * dangle_dR0_z - dangle_dR3_z*dot_product(rkl_x, rkl_y, rkl_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square);

    CoordType dangle_dR2_x = (dot_product(rkl_x, rkl_y, rkl_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square) - 1) * dangle_dR3_x - dangle_dR0_x*dot_product(rij_x, rij_y, rij_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square);
    CoordType dangle_dR2_y = (dot_product(rkl_x, rkl_y, rkl_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square) - 1) * dangle_dR3_y - dangle_dR0_y*dot_product(rij_x, rij_y, rij_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square);
    CoordType dangle_dR2_z = (dot_product(rkl_x, rkl_y, rkl_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square) - 1) * dangle_dR3_z - dangle_dR0_z*dot_product(rij_x, rij_y, rij_z, rkj_x, rkj_y, rkj_z)/(rkj_norm_square);

    CoordType rkj_n = norm(rkj_x, rkj_y, rkj_z);
    rkj_x /= rkj_n;
    rkj_y /= rkj_n;
    rkj_z /= rkj_n;

    CoordType y = dot_product(n3_x, n3_y, n3_z, rkj_x, rkj_y, rkj_z);
    CoordType x = dot_product(n1_x, n1_y, n1_z, n2_x, n2_y, n2_z);
    CoordType angle = atan2(y, x);

    // CoordType sign_angle = sign(timemachine::dot_product(rkj_x, rkj_y, rkj_z, n3_x, n3_y, n3_z));
    // CoordType angle = sign_angle*acos(timemachine::dot_product(n1_x, n1_y, n1_z, n2_x, n2_y, n2_z)/(bot));
    OutType prefactor = k*sin(period*angle - phase)*period;

    grads[0*NDIMS+0] = dangle_dR0_x * prefactor;
    grads[0*NDIMS+1] = dangle_dR0_y * prefactor;
    grads[0*NDIMS+2] = dangle_dR0_z * prefactor;

    grads[1*NDIMS+0] = dangle_dR1_x * prefactor;
    grads[1*NDIMS+1] = dangle_dR1_y * prefactor;
    grads[1*NDIMS+2] = dangle_dR1_z * prefactor;

    grads[2*NDIMS+0] = dangle_dR2_x * prefactor;
    grads[2*NDIMS+1] = dangle_dR2_y * prefactor;
    grads[2*NDIMS+2] = dangle_dR2_z * prefactor;

    grads[3*NDIMS+0] = dangle_dR3_x * prefactor;
    grads[3*NDIMS+1] = dangle_dR3_y * prefactor;
    grads[3*NDIMS+2] = dangle_dR3_z * prefactor;

    return k*(1+cos(period*angle - phase));

}

template<typename RealType, int NDIMS>
void __global__ k_periodic_torsion_derivatives(
    const int num_atoms,    // n
    const RealType *coords, // [n, 3]
    const RealType *params, // [p,]
    const int num_angles,    // a
    const int *angle_idxs,   // [a, 2]
    const int *param_idxs,   // [a, 2]
    RealType *E,             // [,] or null
    RealType *dE_dx,         // [n,3] or null
    RealType *d2E_dx2,       // [C, n, 3, n, 3] or null, hessian
    // parameters used for computing derivatives
    const int num_dp,        // dp, number of parameters we're differentiating w.r.t. 
    const int *param_gather_idxs, // [p,] if -1, then we discard
    RealType *dE_dp,         // [C, dp,] or null
    RealType *d2E_dxdp       // [C, dp, n, 3] or null
) {

    const auto conf_idx = blockIdx.z;
    const int N = num_atoms;
    const int DP = num_dp;
    const auto a_idx = blockDim.x*blockIdx.x + threadIdx.x;

    if(a_idx >= num_angles) {
        return;
    }

    int atom_0_idx = angle_idxs[a_idx*4+0];
    int atom_1_idx = angle_idxs[a_idx*4+1];
    int atom_2_idx = angle_idxs[a_idx*4+2];
    int atom_3_idx = angle_idxs[a_idx*4+3];

    const int indices[12] = {
        atom_0_idx*NDIMS+0,
        atom_0_idx*NDIMS+1,
        atom_0_idx*NDIMS+2,
        atom_1_idx*NDIMS+0,
        atom_1_idx*NDIMS+1,
        atom_1_idx*NDIMS+2,
        atom_2_idx*NDIMS+0,
        atom_2_idx*NDIMS+1,
        atom_2_idx*NDIMS+2,
        atom_3_idx*NDIMS+0,
        atom_3_idx*NDIMS+1,
        atom_3_idx*NDIMS+2
    };


    RealType k = params[param_idxs[a_idx*3+0]];
    RealType phase = params[param_idxs[a_idx*3+1]];
    RealType period = params[param_idxs[a_idx*3+2]];

    RealType xs[12];
    for(int i=0; i < 12; i++) {
        xs[i] = coords[conf_idx*N*NDIMS+indices[i]];
    }

    RealType ps[3] = {k, phase, period};
    RealType dxs[4*NDIMS] = {0};
    RealType energy = torsion_gradient<RealType, RealType, RealType, NDIMS>(xs, ps, dxs);

    if(E) {
        atomicAdd(E + conf_idx, energy);        
    }

    int full_indices[4*NDIMS];
    for(int a=0; a < 4; a++) {
        auto atom_i_idx = angle_idxs[a_idx*4+a]; // not n dims since its an index over 3 atoms.
        for(int d=0; d < NDIMS; d++) {
            full_indices[a*NDIMS+d] = atom_i_idx*NDIMS+d;
        }
    }

    if(dE_dx) {
        for(int i=0; i < 4*NDIMS; i++) {
            atomicAdd(dE_dx + conf_idx*N*NDIMS + full_indices[i], dxs[i]);
        }        
    }

    const RealType step = 1e-7;

    if(d2E_dx2) {
        Surreal<RealType> cxs[12];
        #pragma unroll
        for(int i=0; i < 12; i++) {
            cxs[i] = xs[i];
        }
        for(int j=0; j < 12; j++) {
            cxs[j].imag = step;
            Surreal<RealType> dcxs[4*NDIMS] = {0};
            torsion_gradient<Surreal<RealType>, RealType, Surreal<RealType>, NDIMS >(cxs, ps, dcxs);
            #pragma unroll
            for(int k=0; k < 4*NDIMS; k++) {
                atomicAdd(d2E_dx2 + conf_idx*N*NDIMS*N*NDIMS + indices[j]*N*NDIMS + full_indices[k], dcxs[k].imag / step);
            }
            cxs[j].imag = 0;
        }
    }

    if(dE_dp || d2E_dxdp) {
        Surreal<RealType> cps[3] = {
            ps[0],
            ps[1],
            ps[2]
        };
        for(int j=0; j < 3; j++) {
            int gp_idx = param_gather_idxs[param_idxs[a_idx*3+j]];
            if(gp_idx < 0) {
                continue;
            }
            cps[j].imag = step;
            Surreal<RealType> dcxs[4*NDIMS] = {0};
            Surreal<RealType> denergy = torsion_gradient<RealType, Surreal<RealType>, Surreal<RealType>, NDIMS >(xs, cps, dcxs);
            if(dE_dp) {
                atomicAdd(dE_dp + conf_idx*DP + gp_idx, denergy.imag/step);
            }

            if(d2E_dxdp) {
                #pragma unroll
                for(int k=0; k < 4*NDIMS; k++) {
                    atomicAdd(d2E_dxdp + conf_idx*DP*N*NDIMS + gp_idx*N*NDIMS + full_indices[k], dcxs[k].imag / step);
                }                
            }
            cps[j].imag = 0.0;
        }
    }

}

