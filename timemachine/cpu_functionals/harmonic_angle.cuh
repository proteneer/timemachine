#pragma once

// #include <cstdio>
#include "surreal.cuh"

template <typename CoordType, typename ParamType, typename OutType> 
inline __device__ void gradient(
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

    CoordType dij = nij;
    CoordType djk = njk;

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


}

template <typename NumericType>
__global__ void harmonic_angle_total_derivative(
    const NumericType *coords,
    const NumericType *params,
    const int *global_param_idxs,
    const int *param_idxs,
    const int *angle_idxs,
    NumericType *energy_out,
    NumericType *grad_out,
    NumericType *hessian_out,
    NumericType *mp_out,
    int N,
    int A) {

    auto a_idx = blockDim.x*blockIdx.x + threadIdx.x;

    if(a_idx < A) {

        int atom_0_idx = angle_idxs[a_idx*3+0];
        int atom_1_idx = angle_idxs[a_idx*3+1];
        int atom_2_idx = angle_idxs[a_idx*3+2];

        NumericType xs[9];
        xs[0] = coords[atom_0_idx*3+0];
        xs[1] = coords[atom_0_idx*3+1];
        xs[2] = coords[atom_0_idx*3+2];
        xs[3] = coords[atom_1_idx*3+0];
        xs[4] = coords[atom_1_idx*3+1];
        xs[5] = coords[atom_1_idx*3+2];
        xs[6] = coords[atom_2_idx*3+0];
        xs[7] = coords[atom_2_idx*3+1];
        xs[8] = coords[atom_2_idx*3+2];
        NumericType ps[2];
        ps[0] = params[param_idxs[a_idx*2+0]];
        ps[1] = params[param_idxs[a_idx*2+1]];
        NumericType dxs[9];

        gradient<NumericType, NumericType, NumericType>(xs, ps, dxs);

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

        #pragma unroll
        for(int i=0; i < 9; i++) {
            atomicAdd(grad_out + indices[i], dxs[i]);
        }

        NumericType step = 1e-35;



        Surreal<NumericType> cps[2] = {
            ps[0],
            ps[1]
        };

        // #pragma unroll
        for(int j=0; j < 2; j++) {
            cps[j].deriv = step;
            Surreal<NumericType> dcxs[9];
            gradient<NumericType, Surreal<NumericType>, Surreal<NumericType> >(xs, cps, dcxs);
            int gp_idx = global_param_idxs[param_idxs[a_idx*2+j]];
            #pragma unroll
            for(int k=0; k < 9; k++) {
                atomicAdd(mp_out + gp_idx*N*3 + indices[k], dcxs[k].deriv / step);
            }
            cps[j].deriv = 0.0;
        }



        // #pragma unroll
        for(int j=0; j < 9; j++) {

            Surreal<NumericType> cxs[9] = {
                coords[atom_0_idx*3+0],
                coords[atom_0_idx*3+1],
                coords[atom_0_idx*3+2],
                coords[atom_1_idx*3+0],
                coords[atom_1_idx*3+1],
                coords[atom_1_idx*3+2],
                coords[atom_2_idx*3+0],
                coords[atom_2_idx*3+1],
                coords[atom_2_idx*3+2]
            };
            cxs[j].deriv = step;
            Surreal<NumericType> dcxs[9];
            gradient<Surreal<NumericType>, NumericType, Surreal<NumericType> >(cxs, ps, dcxs);
            #pragma unroll
            for(int k=0; k < 9; k++) {
                atomicAdd(hessian_out + indices[j]*N*3 + indices[k], dcxs[k].deriv / step);
            }
            cxs[j].deriv = 0;
        }


    }

}