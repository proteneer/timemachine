#pragma once

#include "surreal.cuh"

template <typename CoordType, typename ParamType, typename OutType> 
inline __device__ OutType harmonic_angle_gradient(
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
    ParamType ka = params[0];
    ParamType a0 = params[1];

    CoordType vij_x = x1 - x0;
    CoordType vij_y = y1 - y0;
    CoordType vij_z = z1 - z0;


    CoordType vjk_x = x1 - x2;
    CoordType vjk_y = y1 - y2;
    CoordType vjk_z = z1 - z2;

    CoordType nij = sqrt(vij_x*vij_x + vij_y*vij_y + vij_z*vij_z);
    CoordType njk = sqrt(vjk_x*vjk_x + vjk_y*vjk_y + vjk_z*vjk_z);

    CoordType nijk = nij*njk;
    CoordType n3ij = nij*nij*nij;
    CoordType n3jk = njk*njk*njk;

    CoordType top = vij_x*vjk_x + vij_y*vjk_y + vij_z*vjk_z;
    // CoordType dijk = dij*djk;

    ParamType cos_a0 = cos(a0);
    OutType delta = top/nijk - cos(a0);

    grads[0*3+0] = ka*delta*((-x0 + x1)*(top)/(n3ij*njk) + (-x1 + x2)/(nijk));
    grads[1*3+0] = ka*delta*((x0 - x1)*(top)/(n3ij*njk) + (-x1 + x2)*(top)/(nij*n3jk) + (-x0 + 2.0*x1 - x2)/(nijk));
    grads[2*3+0] = ka*((x0 - x1)/(nijk) + (x1 - x2)*(top)/(nij*n3jk))*delta;
    grads[0*3+1] = ka*delta*((-y0 + y1)*(top)/(n3ij*njk) + (-y1 + y2)/(nijk));
    grads[1*3+1] = ka*delta*((y0 - y1)*(top)/(n3ij*njk) + (-y1 + y2)*(top)/(nij*n3jk) + (-y0 + 2.0*y1 - y2)/(nijk));
    grads[2*3+1] = ka*((y0 - y1)/(nijk) + (y1 - y2)*(top)/(nij*n3jk))*delta;
    grads[0*3+2] = ka*delta*((-z0 + z1)*(top)/(n3ij*njk) + (-z1 + z2)/(nijk));
    grads[1*3+2] = ka*delta*((z0 - z1)*(top)/(n3ij*njk) + (-z1 + z2)*(top)/(nij*n3jk) + (-z0 + 2.0*z1 - z2)/(nijk));
    grads[2*3+2] = ka*((z0 - z1)/(nijk) + (z1 - z2)*(top)/(nij*n3jk))*delta;

    return ka/2*(delta*delta);

}

template<typename RealType>
void __global__ k_harmonic_angle_derivatives(
    const int num_atoms,     // n, number of atoms
    const RealType *coords,  // [C, n, 3]
    const RealType *params,  // [p,]
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

    int atom_0_idx = angle_idxs[a_idx*3+0];
    int atom_1_idx = angle_idxs[a_idx*3+1];
    int atom_2_idx = angle_idxs[a_idx*3+2];

    const int indices[9] = {
        atom_0_idx*3+0,
        atom_0_idx*3+1,
        atom_0_idx*3+2,
        atom_1_idx*3+0,
        atom_1_idx*3+1,
        atom_1_idx*3+2,
        atom_2_idx*3+0,
        atom_2_idx*3+1,
        atom_2_idx*3+2
    };


    RealType xs[9];
    #pragma unroll
    for(int i=0; i < 9; i++) {
        xs[i] = coords[conf_idx*N*3 + indices[i]];
    }

    RealType ps[2];
    ps[0] = params[param_idxs[a_idx*2+0]];
    ps[1] = params[param_idxs[a_idx*2+1]];
    RealType dxs[9];

    RealType energy = harmonic_angle_gradient<RealType, RealType, RealType>(xs, ps, dxs);

    if(E) {
        atomicAdd(E + conf_idx , energy);        
    }

    if(dE_dx) {
        for(int i=0; i < 9; i++) {
            atomicAdd(dE_dx + conf_idx*N*3 + indices[i], dxs[i]);
        }        
    }

    const RealType step = 1e-7;

    if(d2E_dx2) {
        Surreal<RealType> cxs[9];
        #pragma unroll
        for(int i=0; i < 9; i++) {
            cxs[i] = xs[i];
        }
        for(int j=0; j < 9; j++) {
            cxs[j].imag = step;
            Surreal<RealType> dcxs[9];
            harmonic_angle_gradient<Surreal<RealType>, RealType, Surreal<RealType> >(cxs, ps, dcxs);
            #pragma unroll
            for(int k=0; k < 9; k++) {
                atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + indices[j]*N*3 + indices[k], dcxs[k].imag / step);
            }
            cxs[j].imag = 0;
        }
    }

    if(dE_dp || d2E_dxdp) {

        Surreal<RealType> cps[2] = {
            ps[0],
            ps[1]
        };

        for(int j=0; j < 2; j++) {
            cps[j].imag = step;
            Surreal<RealType> dcxs[9];
            Surreal<RealType> denergy = harmonic_angle_gradient<RealType, Surreal<RealType>, Surreal<RealType> >(xs, cps, dcxs);
            int gp_idx = param_gather_idxs[param_idxs[a_idx*2+j]];
            if(dE_dp && gp_idx >= 0) {
                atomicAdd(dE_dp + conf_idx*DP + gp_idx, denergy.imag/step);
            }
            if(d2E_dxdp && gp_idx >= 0) {
                #pragma unroll
                for(int k=0; k < 9; k++) {
                    atomicAdd(d2E_dxdp + conf_idx*DP*N*3 + gp_idx*N*3 + indices[k], dcxs[k].imag / step);
                }                
            }
            cps[j].imag = 0.0;
        }
    }

}
