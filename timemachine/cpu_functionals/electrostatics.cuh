#pragma once

#include "kernel_utils.cuh"
/*
    
(ytz): The nonbonded electrostatics algorithm. 

The algorithm takes advantage of warp-level programming. The fundamental unit of
computation is a block of warp_size x warp_size atom interactions. Suppose that
warp_size is equal to 4, we then compute forces, eneriges, mixed_partials, and
diagonal* components of the hessian by using staggered indexing. In the diagram
below, the first round we compute the 0s, accumulating as appropriate, in the
second second compute the 1s, etc. until warp_size-1 is done. Along the way, we
use shuffle intrinsics to swap memory as appropriate.
          
       x0 x1 x2 x3
t0  x0  0  1  2  3
t1  x1  3  0  1  2
t2  x2  2  3  0  1
t3  x3  1  2  3  0 

For the off-diagonal* elements of tne hessian.

* Recall that the hessian of an interaction E(x0, y0, z0, x1, y1, z1) is formally
defined as d^2E/dxi dxj, which has two diagonal 3x3 self-interaction terms, and
two off-diagonal 3x3 interaction terms.



Two additional improvements can be made;

1. Use NVRTC to JIT the number of atoms, N
2. Modify scale_matrix so the bonded scaling terms are done using a bonded kernel.

*/



template<typename NumericType>
__global__ void electrostatics_total_derivative(
    const NumericType *coords,
    const NumericType *params, // change to int later?
    const int *global_param_idxs, // change to int later?
    const int *param_idxs,
    const NumericType *scale_matrix,
    NumericType *energy_out,
    NumericType *grad_out,
    NumericType *hessian_out,
    NumericType *mp_out,
    int N) {

    const int n_atoms = N;

    auto i_idx = blockDim.x*blockIdx.x + threadIdx.x;
    
    // this points to some place nebulous
    NumericType *mp_out_qi;
    
    NumericType x0, y0, z0, q0;

    if(i_idx >= n_atoms) {
        x0 = 0.0;
        y0 = 0.0;
        z0 = 0.0;
        q0 = 0.0;
    } else {
        x0 = coords[i_idx*3+0];
        y0 = coords[i_idx*3+1];
        z0 = coords[i_idx*3+2];
        q0 = params[param_idxs[i_idx]];
        mp_out_qi = mp_out + global_param_idxs[param_idxs[i_idx]]*n_atoms*3;
    }

    NumericType grad_dx = 0;
    NumericType grad_dy = 0;
    NumericType grad_dz = 0;

    NumericType hess_xx = 0;
    NumericType hess_yx = 0;
    NumericType hess_yy = 0;
    NumericType hess_zx = 0;
    NumericType hess_zy = 0;
    NumericType hess_zz = 0;

    int num_y_tiles = blockIdx.x + 1;

    for(int tile_y_idx = 0; tile_y_idx < num_y_tiles; tile_y_idx++) {

        NumericType x1, y1, z1, q1;
        NumericType shfl_grad_dx = 0;
        NumericType shfl_grad_dy = 0;
        NumericType shfl_grad_dz = 0;

        NumericType shfl_hess_xx = 0;
        NumericType shfl_hess_yx = 0;
        NumericType shfl_hess_yy = 0;
        NumericType shfl_hess_zx = 0;
        NumericType shfl_hess_zy = 0;
        NumericType shfl_hess_zz = 0;

        // load diagonal elements exactly once, shuffle the rest
        int j_idx = tile_y_idx*WARP_SIZE + threadIdx.x;

        if(j_idx >= n_atoms) {
            x1 = 0.0;
            y1 = 0.0;
            z1 = 0.0;
            q1 = 0.0;
        } else {
            x1 = coords[j_idx*3+0];
            y1 = coords[j_idx*3+1];
            z1 = coords[j_idx*3+2];
            q1 = params[param_idxs[j_idx]];
        }

        // off diagonal
        // iterate over a block of i's because we improve locality of writes to off diagonal elements
        #pragma unroll 4
        for(int round=0; round < WARP_SIZE; round++) {
            NumericType xi = __shfl_sync(0xffffffff, x0, round);
            NumericType yi = __shfl_sync(0xffffffff, y0, round);
            NumericType zi = __shfl_sync(0xffffffff, z0, round);
            NumericType qi = __shfl_sync(0xffffffff, q0, round);

            int h_i_idx = blockIdx.x*WARP_SIZE + round;
            int h_j_idx = j_idx;


            if(h_j_idx < h_i_idx && h_i_idx < n_atoms && h_j_idx < n_atoms) {

                NumericType dx = xi - x1;
                NumericType dy = yi - y1;
                NumericType dz = zi - z1;
                NumericType d2x = dx*dx;
                NumericType d2y = dy*dy;
                NumericType d2z = dz*dz;

                NumericType d2ij = d2x + d2y + d2z;
                NumericType dij = gpuSqrt(d2ij);
                NumericType d3ij = d2ij*dij;
                NumericType d5ij = d3ij*d2ij;

                NumericType sij = scale_matrix[h_i_idx*n_atoms + h_j_idx];
                NumericType so4eq01 = sij*ONE_4PI_EPS0*qi*q1;
                NumericType hess_prefactor = so4eq01/d5ij;

                hessian_out[HESS_IDX(h_i_idx, h_j_idx, n_atoms, 0, 0)] += hess_prefactor*(d2ij - 3*d2x);
                hessian_out[HESS_IDX(h_i_idx, h_j_idx, n_atoms, 0, 1)] += -3*hess_prefactor*dx*dy;
                hessian_out[HESS_IDX(h_i_idx, h_j_idx, n_atoms, 0, 2)] += -3*hess_prefactor*dx*dz;
                hessian_out[HESS_IDX(h_i_idx, h_j_idx, n_atoms, 1, 0)] += -3*hess_prefactor*dx*dy;
                hessian_out[HESS_IDX(h_i_idx, h_j_idx, n_atoms, 1, 1)] += hess_prefactor*(d2ij - 3*d2y);
                hessian_out[HESS_IDX(h_i_idx, h_j_idx, n_atoms, 1, 2)] += -3*hess_prefactor*dy*dz;
                hessian_out[HESS_IDX(h_i_idx, h_j_idx, n_atoms, 2, 0)] += -3*hess_prefactor*dx*dz;
                hessian_out[HESS_IDX(h_i_idx, h_j_idx, n_atoms, 2, 1)] += -3*hess_prefactor*dy*dz;
                hessian_out[HESS_IDX(h_i_idx, h_j_idx, n_atoms, 2, 2)] += hess_prefactor*(d2ij - 3*d2z);
            }

        }

        // diagonal elements and mixed partials
        for(int round=0; round < WARP_SIZE; round++) {

            j_idx = tile_y_idx*WARP_SIZE + j_idx % WARP_SIZE;

            if(j_idx < i_idx && i_idx < n_atoms && j_idx < n_atoms) {

                NumericType dx = x0 - x1;
                NumericType dy = y0 - y1;
                NumericType dz = z0 - z1;
                NumericType d2x = dx*dx;
                NumericType d2y = dy*dy;
                NumericType d2z = dz*dz;

                NumericType d2ij = d2x + d2y + d2z;
                NumericType dij = sqrt(d2ij);
                NumericType d3ij = d2ij*dij;
                NumericType inv_d3ij = 1/d3ij;
                NumericType d5ij = d3ij*d2ij;

                NumericType sij = scale_matrix[i_idx*n_atoms + j_idx];
                NumericType so4eq01 = sij*ONE_4PI_EPS0*q0*q1;
                NumericType grad_prefactor = so4eq01*inv_d3ij;
                NumericType hess_prefactor = so4eq01/d5ij;

                grad_dx -= grad_prefactor*dx;
                grad_dy -= grad_prefactor*dy;
                grad_dz -= grad_prefactor*dz;

                shfl_grad_dx += grad_prefactor*dx;
                shfl_grad_dy += grad_prefactor*dy;
                shfl_grad_dz += grad_prefactor*dz;

                NumericType *mp_out_qj = mp_out + global_param_idxs[param_idxs[j_idx]]*n_atoms*3;

                NumericType mp_prefactor = sij*ONE_4PI_EPS0*inv_d3ij;

                NumericType PREFACTOR_QI_GRAD = mp_prefactor*q1;
                NumericType PREFACTOR_QJ_GRAD = mp_prefactor*q0;

                // (ytz): We can optimize this further by assuming that every particle has
                // its own parameter, there by forming a (2N*N*3 mixed partial buffer), followed
                // by a segmented sum to accumuluate the shared results. For the sake of simplicity,
                // we use atomicAdds for now.
                atomicAdd(mp_out_qi + i_idx*3 + 0, PREFACTOR_QI_GRAD * (-dx));
                atomicAdd(mp_out_qi + i_idx*3 + 1, PREFACTOR_QI_GRAD * (-dy));
                atomicAdd(mp_out_qi + i_idx*3 + 2, PREFACTOR_QI_GRAD * (-dz));
                atomicAdd(mp_out_qi + j_idx*3 + 0, PREFACTOR_QI_GRAD * (dx));
                atomicAdd(mp_out_qi + j_idx*3 + 1, PREFACTOR_QI_GRAD * (dy));
                atomicAdd(mp_out_qi + j_idx*3 + 2, PREFACTOR_QI_GRAD * (dz));

                atomicAdd(mp_out_qj + i_idx*3 + 0, PREFACTOR_QJ_GRAD * (-dx));
                atomicAdd(mp_out_qj + i_idx*3 + 1, PREFACTOR_QJ_GRAD * (-dy));
                atomicAdd(mp_out_qj + i_idx*3 + 2, PREFACTOR_QJ_GRAD * (-dz));
                atomicAdd(mp_out_qj + j_idx*3 + 0, PREFACTOR_QJ_GRAD * (dx));
                atomicAdd(mp_out_qj + j_idx*3 + 1, PREFACTOR_QJ_GRAD * (dy));
                atomicAdd(mp_out_qj + j_idx*3 + 2, PREFACTOR_QJ_GRAD * (dz));

                // compute lower triangular elements
                hess_xx += hess_prefactor*(-d2ij + 3*d2x);
                hess_yx += 3*hess_prefactor*dx*dy;
                hess_yy += hess_prefactor*(-d2ij + 3*d2y);
                hess_zx += 3*hess_prefactor*dx*dz;
                hess_zy += 3*hess_prefactor*dy*dz;
                hess_zz += hess_prefactor*(-d2ij + 3*d2z);

                shfl_hess_xx += hess_prefactor*(-d2ij + 3*d2x);
                shfl_hess_yx += 3*hess_prefactor*dx*dy;
                shfl_hess_yy += hess_prefactor*(-d2ij + 3*d2y);
                shfl_hess_zx += 3*hess_prefactor*dx*dz;
                shfl_hess_zy += 3*hess_prefactor*dy*dz;
                shfl_hess_zz += hess_prefactor*(-d2ij + 3*d2z);

            }

            int srcLane = (threadIdx.x + 1) % WARP_SIZE;

            x1 = __shfl_sync(0xffffffff, x1, srcLane);
            y1 = __shfl_sync(0xffffffff, y1, srcLane);
            z1 = __shfl_sync(0xffffffff, z1, srcLane);
            q1 = __shfl_sync(0xffffffff, q1, srcLane);

            shfl_grad_dx = __shfl_sync(0xffffffff, shfl_grad_dx, srcLane);
            shfl_grad_dy = __shfl_sync(0xffffffff, shfl_grad_dy, srcLane);
            shfl_grad_dz = __shfl_sync(0xffffffff, shfl_grad_dz, srcLane);

            shfl_hess_xx = __shfl_sync(0xffffffff, shfl_hess_xx, srcLane);
            shfl_hess_yx = __shfl_sync(0xffffffff, shfl_hess_yx, srcLane);
            shfl_hess_yy = __shfl_sync(0xffffffff, shfl_hess_yy, srcLane);
            shfl_hess_zx = __shfl_sync(0xffffffff, shfl_hess_zx, srcLane);
            shfl_hess_zy = __shfl_sync(0xffffffff, shfl_hess_zy, srcLane);
            shfl_hess_zz = __shfl_sync(0xffffffff, shfl_hess_zz, srcLane);

            j_idx += 1;

        }

        int target_idx = tile_y_idx*WARP_SIZE + j_idx % WARP_SIZE;

        if(target_idx < n_atoms) {
            atomicAdd(grad_out + target_idx*3 + 0, shfl_grad_dx);
            atomicAdd(grad_out + target_idx*3 + 1, shfl_grad_dy);
            atomicAdd(grad_out + target_idx*3 + 2, shfl_grad_dz);

            atomicAdd(hessian_out + HESS_IDX(target_idx, target_idx, n_atoms, 0, 0), shfl_hess_xx);
            atomicAdd(hessian_out + HESS_IDX(target_idx, target_idx, n_atoms, 1, 0), shfl_hess_yx);
            atomicAdd(hessian_out + HESS_IDX(target_idx, target_idx, n_atoms, 1, 1), shfl_hess_yy);
            atomicAdd(hessian_out + HESS_IDX(target_idx, target_idx, n_atoms, 2, 0), shfl_hess_zx);
            atomicAdd(hessian_out + HESS_IDX(target_idx, target_idx, n_atoms, 2, 1), shfl_hess_zy);
            atomicAdd(hessian_out + HESS_IDX(target_idx, target_idx, n_atoms, 2, 2), shfl_hess_zz);
        }

    }

    if(i_idx < n_atoms) {
        atomicAdd(grad_out + i_idx*3 + 0, grad_dx);
        atomicAdd(grad_out + i_idx*3 + 1, grad_dy);
        atomicAdd(grad_out + i_idx*3 + 2, grad_dz);

        atomicAdd(hessian_out + HESS_IDX(i_idx, i_idx, n_atoms, 0, 0), hess_xx);
        atomicAdd(hessian_out + HESS_IDX(i_idx, i_idx, n_atoms, 1, 0), hess_yx);
        atomicAdd(hessian_out + HESS_IDX(i_idx, i_idx, n_atoms, 1, 1), hess_yy);
        atomicAdd(hessian_out + HESS_IDX(i_idx, i_idx, n_atoms, 2, 0), hess_zx);
        atomicAdd(hessian_out + HESS_IDX(i_idx, i_idx, n_atoms, 2, 1), hess_zy);
        atomicAdd(hessian_out + HESS_IDX(i_idx, i_idx, n_atoms, 2, 2), hess_zz);

    }

};
