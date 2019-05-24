#pragma once

#include "surreal.cuh"

template<typename RealType>
void __global__ k_harmonic_angle_derivatives(
    const int num_atoms,    // n
    const int num_params,   // p
    const RealType *coords, // [n, 3]
    const RealType *params, // [p,]
    const int num_angles,    // a
    const int *angle_idxs,   // [a, 3]
    const int *param_idxs,  // [a, 2]
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

    int atom_0_idx = angle_idxs[a_idx*3+0];
    int atom_1_idx = angle_idxs[a_idx*3+1];
    int atom_2_idx = angle_idxs[a_idx*3+2];

    RealType rx0 = coords[conf_idx*num_atoms*3+atom_0_idx*3+0];
    RealType ry0 = coords[conf_idx*num_atoms*3+atom_0_idx*3+1];
    RealType rz0 = coords[conf_idx*num_atoms*3+atom_0_idx*3+2];

    RealType rx1 = coords[conf_idx*num_atoms*3+atom_1_idx*3+0];
    RealType ry1 = coords[conf_idx*num_atoms*3+atom_1_idx*3+1];
    RealType rz1 = coords[conf_idx*num_atoms*3+atom_1_idx*3+2];

    RealType rx2 = coords[conf_idx*num_atoms*3+atom_2_idx*3+0];
    RealType ry2 = coords[conf_idx*num_atoms*3+atom_2_idx*3+1];
    RealType rz2 = coords[conf_idx*num_atoms*3+atom_2_idx*3+2];

    RealType ix0 = 0;
    RealType iy0 = 0;
    RealType iz0 = 0;

    RealType ix1 = 0;
    RealType iy1 = 0;
    RealType iz1 = 0;

    RealType ix2 = 0;
    RealType iy2 = 0;
    RealType iz2 = 0;

    RealType rka = params[param_idxs[a_idx*2+0]];
    RealType ra0 = params[param_idxs[a_idx*2+1]];

    RealType ika = 0;
    RealType ia0 = 0;

    const auto dp_idx = blockIdx.y;
    const auto num_dp = gridDim.y;

    // (ytz): this a complex step size, not a standard finite difference step size.
    // the error decays quadratically as opposed to linearly w.r.t. step size.
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
        }

        // complex-step parameters 
        if(param_idxs[a_idx*2+0] == dp_idxs[dp_idx]){
            ika = step_size;
        }
        if(param_idxs[a_idx*2+1] == dp_idxs[dp_idx]){
            ia0 = step_size;
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

    Surreal<RealType> ka(rka, ika);
    Surreal<RealType> a0(ra0, ia0);

    Surreal<RealType> vij_x = x1 - x0;
    Surreal<RealType> vij_y = y1 - y0;
    Surreal<RealType> vij_z = z1 - z0;

    Surreal<RealType> vjk_x = x1 - x2;
    Surreal<RealType> vjk_y = y1 - y2;
    Surreal<RealType> vjk_z = z1 - z2;

    Surreal<RealType> nij = sqrt(vij_x*vij_x + vij_y*vij_y + vij_z*vij_z);
    Surreal<RealType> njk = sqrt(vjk_x*vjk_x + vjk_y*vjk_y + vjk_z*vjk_z);

    Surreal<RealType> nijk = nij*njk;
    Surreal<RealType> n3ij = nij*nij*nij;
    Surreal<RealType> n3jk = njk*njk*njk;

    Surreal<RealType> top = vij_x*vjk_x + vij_y*vjk_y + vij_z*vjk_z;

    Surreal<RealType> cos_a0 = cos(a0);
    Surreal<RealType> delta = top/nijk - cos(a0);

    Surreal<RealType> atom_0_grad_x = ka*delta*((-x0 + x1)*(top)/(n3ij*njk) + (-x1 + x2)/(nijk));
    Surreal<RealType> atom_0_grad_y = ka*delta*((-y0 + y1)*(top)/(n3ij*njk) + (-y1 + y2)/(nijk));
    Surreal<RealType> atom_0_grad_z = ka*delta*((-z0 + z1)*(top)/(n3ij*njk) + (-z1 + z2)/(nijk));

    Surreal<RealType> atom_1_grad_x = ka*delta*((x0 - x1)*(top)/(n3ij*njk) + (-x1 + x2)*(top)/(nij*n3jk) + (-x0 + 2.0*x1 - x2)/(nijk));
    Surreal<RealType> atom_1_grad_y = ka*delta*((y0 - y1)*(top)/(n3ij*njk) + (-y1 + y2)*(top)/(nij*n3jk) + (-y0 + 2.0*y1 - y2)/(nijk));
    Surreal<RealType> atom_1_grad_z = ka*delta*((z0 - z1)*(top)/(n3ij*njk) + (-z1 + z2)*(top)/(nij*n3jk) + (-z0 + 2.0*z1 - z2)/(nijk));

    Surreal<RealType> atom_2_grad_x = ka*((x0 - x1)/(nijk) + (x1 - x2)*(top)/(nij*n3jk))*delta;
    Surreal<RealType> atom_2_grad_y = ka*((y0 - y1)/(nijk) + (y1 - y2)*(top)/(nij*n3jk))*delta;
    Surreal<RealType> atom_2_grad_z = ka*((z0 - z1)/(nijk) + (z1 - z2)*(top)/(nij*n3jk))*delta;

    Surreal<RealType> energy = ka/2*(delta*delta);

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
    }

}
