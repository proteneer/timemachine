#include "nonbonded_gpu.hpp"
#include "gpu_utils.cuh"


#include <ctime>
#include <iostream>

#define ONE_4PI_EPS0 138.935456

namespace timemachine {



inline __device__ float gpuSqrt(float arg) {
  return sqrtf(arg);
}

inline __device__ double gpuSqrt(double arg) {
  return sqrt(arg);
}


/*
(ytz): The shuffling nonbonded algorithm

    wi warp index
    ri round i


    w0   |  w1   |  w2    
  0 1 2 3|4 5 6 7|8 9 0 1
  -----------------------
0 0 1 2 3|0 1 2 3|0 1 2 3
1 x 0 1 2|3 0 1 2|3 0 1 2
2 x x 0 1|2 3 0 1|2 3 0 1
3 x x x 0|1 2 3 0|1 2 3 0
- -----------------------
4 x x x x|0 1 2 3|0 1 2 3
5 x x x x|x 0 1 2|3 0 1 2
6 x x x x|x x 0 1|2 3 0 1
7 x x x x|x x x 0|1 2 3 0
- -----------------------
8 x x x x|x x x x|0 1 2 3
9 x x x x|x x x x|x 0 1 2
0 x x x x|x x x x|x x 0 1
1 x x x x|x x x x|x x x 0


t |0 1 2 3|0 1 2 3|0 1 2 3|0 1 2 3
----------------------------------
w0|  r0   |  r1   |  r2   |  r3
--|-------|-------|-------|-------
i |0 1 2 3|0 1 2 3|0 1 2 3|0 1 2 3
j |0 1 2 3|1 2 3 0|2 3 0 1|3 0 1 2


t |0 1 2 3|0 1 2 3|0 1 2 3|0 1 2 3
----------------------------------
w1|  r0   |  r1   |  r2   |  r3
--|-------|-------|-------|-------
i |4 5 6 7|4 5 6 7|4 5 6 7|4 5 6 7
j |0 1 2 3|1 2 3 0|2 3 0 1|3 0 1 2 


warp 1 round 0







*/
#define N_HARDCODE 2489
#define WARP_SIZE 32

// should we change layout to 3N x 3N to improve coalesced reads and writes?
// probably *especially* important for hessians.

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
    int P,
    int N) {

    const int n_atoms = N;
    const int N3 = n_atoms*3;

    auto i_idx = blockDim.x*blockIdx.x + threadIdx.x;

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
    }

    NumericType grad_dx = 0;
    NumericType grad_dy = 0;
    NumericType grad_dz = 0;

    NumericType hess_xx = 0;
    NumericType hess_xy = 0;
    NumericType hess_xz = 0;
    NumericType hess_yx = 0;
    NumericType hess_yy = 0;
    NumericType hess_yz = 0;
    NumericType hess_zx = 0;
    NumericType hess_zy = 0;
    NumericType hess_zz = 0;

    // int tile_x_idx = (n_atoms + WARP_SIZE - 1)/WARP_SIZE;
    // int num_y_tiles = tile_x_idx;

    // int tile_x_idx = (n_atoms + WARP_SIZE - 1)/WARP_SIZE;
    int num_y_tiles = blockIdx.x + 1;

    // printf("block %d NUM_Y_TILES %d\n", blockIdx.x, num_y_tiles);

    for(int tile_y_idx = 0; tile_y_idx < num_y_tiles; tile_y_idx++) {

        NumericType x1, y1, z1, q1;
        NumericType shfl_grad_dx = 0;
        NumericType shfl_grad_dy = 0;
        NumericType shfl_grad_dz = 0;

        NumericType shfl_hess_xx = 0;
        NumericType shfl_hess_xy = 0;
        NumericType shfl_hess_xz = 0;
        NumericType shfl_hess_yx = 0;
        NumericType shfl_hess_yy = 0;
        NumericType shfl_hess_yz = 0;
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


        // if(blockIdx.x == 0) {
            // printf("blockIdx.x %d threadIdx.x %d | I_IDX J_IDX  %d %d | xyz %f %f %f | shfl accum %f %f %f \n", blockIdx.x, threadIdx.x, i_idx, j_idx, x1, y1, z1, shfl_grad_dx, shfl_grad_dy, shfl_grad_dz);
        // }

        // printf("tid x1 y1 z1, %f %f %f\n", x0, y0, z0);

        // #pragma unroll 4
        for(int round=0; round < WARP_SIZE; round++) {

            j_idx = tile_y_idx*WARP_SIZE + j_idx % WARP_SIZE;

            // if(blockIdx.x == 1) {
            //     printf("round blockIdx.x, %d %d |I_IDX J_IDX  %d %d\n", round, blockIdx.x, i_idx, j_idx);
            // }

            NumericType dx = x0 - x1;
            NumericType dy = y0 - y1;
            NumericType dz = z0 - z1;
            NumericType d2x = dx*dx;
            NumericType d2y = dy*dy;
            NumericType d2z = dz*dz;

            NumericType d2ij = d2x + d2y + d2z;
            NumericType dij = sqrt(d2ij);
            NumericType d3ij = d2ij*dij;
            NumericType d5ij = d3ij*d2ij;

            // printf("round %d tid %d dij %f x1 %f y1 %f z1%f\n", round, threadIdx.x, dij, x1, y1, z1);

            // todo: optimize
            NumericType sij = 0;
            if(i_idx < n_atoms && j_idx < n_atoms) {
                sij = scale_matrix[i_idx*n_atoms + j_idx];
                // sij = 0.5f;
            } else {
                sij = 0;
            }


            NumericType so4eq01 = sij*ONE_4PI_EPS0*q0*q1;

            // atom_nrg += so4eq01/dij;
            // compute gradients.

            NumericType grad_prefactor = so4eq01/d3ij;
            NumericType hess_prefactor = so4eq01/d5ij;


            if(j_idx < i_idx && i_idx < n_atoms && j_idx < n_atoms) {

                // printf("computing gradient between i_idx %d j_idx %d\n", i_idx, j_idx);

                grad_dx -= grad_prefactor*dx;
                grad_dy -= grad_prefactor*dy;
                grad_dz -= grad_prefactor*dz;

                shfl_grad_dx += grad_prefactor*dx;
                shfl_grad_dy += grad_prefactor*dy;
                shfl_grad_dz += grad_prefactor*dz;


                NumericType prefactor = hess_prefactor;

                // compute lower triangular

                hess_xx += hess_prefactor*(-d2ij + 3*d2x);
                // hess_xy += 3*hess_prefactor*dx*dy;
                // hess_xz += 3*hess_prefactor*dx*dz;
                hess_yx += 3*hess_prefactor*dx*dy;
                hess_yy += hess_prefactor*(-d2ij + 3*d2y);
                // hess_yz += 3*hess_prefactor*dy*dz;
                hess_zx += 3*hess_prefactor*dx*dz;
                hess_zy += 3*hess_prefactor*dy*dz;
                hess_zz += hess_prefactor*(-d2ij + 3*d2z);

                const size_t x_dim = 0;
                const size_t y_dim = 1;
                const size_t z_dim = 2;

                shfl_hess_xx += hess_prefactor*(-d2ij + 3*d2x);
                // shfl_hess_xy += 3*hess_prefactor*dx*dy;
                // shfl_hess_xz += 3*hess_prefactor*dx*dz;
                shfl_hess_yx += 3*hess_prefactor*dx*dy;
                shfl_hess_yy += hess_prefactor*(-d2ij + 3*d2y);
                // shfl_hess_yz += 3*hess_prefactor*dy*dz;
                shfl_hess_zx += 3*hess_prefactor*dx*dz;
                shfl_hess_zy += 3*hess_prefactor*dy*dz;
                shfl_hess_zz += hess_prefactor*(-d2ij + 3*d2z);

                // hessian_out[0*N*3*N + i_idx*3*N + 0*N + j_idx] += prefactor*(d2ij - 3*d2x);
                // hessian_out[0*N*3*N + i_idx*3*N + 1*N + j_idx] += -3*prefactor*dx*dy;
                // hessian_out[0*N*3*N + i_idx*3*N + 2*N + j_idx] += -3*prefactor*dx*dz;
                // hessian_out[1*N*3*N + i_idx*3*N + 0*N + j_idx] += -3*prefactor*dx*dy;
                // hessian_out[1*N*3*N + i_idx*3*N + 1*N + j_idx] += prefactor*(d2ij - 3*d2y);
                // hessian_out[1*N*3*N + i_idx*3*N + 2*N + j_idx] += -3*prefactor*dy*dz;
                // hessian_out[2*N*3*N + i_idx*3*N + 0*N + j_idx] += -3*prefactor*dx*dz;
                // hessian_out[2*N*3*N + i_idx*3*N + 1*N + j_idx] += -3*prefactor*dy*dz;
                // hessian_out[2*N*3*N + i_idx*3*N + 2*N + j_idx] += prefactor*(d2ij - 3*d2z);

                hessian_out[i_idx * 3 * N * 3 + x_dim * N * 3 + j_idx * 3 + x_dim] += prefactor*(d2ij - 3*d2x);
                hessian_out[i_idx * 3 * N * 3 + x_dim * N * 3 + j_idx * 3 + y_dim] += -3*prefactor*dx*dy;
                hessian_out[i_idx * 3 * N * 3 + x_dim * N * 3 + j_idx * 3 + z_dim] += -3*prefactor*dx*dz;
                hessian_out[i_idx * 3 * N * 3 + y_dim * N * 3 + j_idx * 3 + x_dim] += -3*prefactor*dx*dy;
                hessian_out[i_idx * 3 * N * 3 + y_dim * N * 3 + j_idx * 3 + y_dim] += prefactor*(d2ij - 3*d2y);
                hessian_out[i_idx * 3 * N * 3 + y_dim * N * 3 + j_idx * 3 + z_dim] += -3*prefactor*dy*dz;
                hessian_out[i_idx * 3 * N * 3 + z_dim * N * 3 + j_idx * 3 + x_dim] += -3*prefactor*dx*dz;
                hessian_out[i_idx * 3 * N * 3 + z_dim * N * 3 + j_idx * 3 + y_dim] += -3*prefactor*dy*dz;
                hessian_out[i_idx * 3 * N * 3 + z_dim * N * 3 + j_idx * 3 + z_dim] += prefactor*(d2ij - 3*d2z);


                // j < i
                // hessian_out[j_idx * 3 * N * 3 + x_dim * N * 3 + i_idx * 3 + x_dim] += prefactor*(d2ij - 3*d2x);
                // hessian_out[j_idx * 3 * N * 3 + x_dim * N * 3 + i_idx * 3 + y_dim] += -3*prefactor*dx*dy;
                // hessian_out[j_idx * 3 * N * 3 + x_dim * N * 3 + i_idx * 3 + z_dim] += -3*prefactor*dx*dz;
                // hessian_out[j_idx * 3 * N * 3 + y_dim * N * 3 + i_idx * 3 + x_dim] += -3*prefactor*dx*dy;
                // hessian_out[j_idx * 3 * N * 3 + y_dim * N * 3 + i_idx * 3 + y_dim] += prefactor*(d2ij - 3*d2y);
                // hessian_out[j_idx * 3 * N * 3 + y_dim * N * 3 + i_idx * 3 + z_dim] += -3*prefactor*dy*dz;
                // hessian_out[j_idx * 3 * N * 3 + z_dim * N * 3 + i_idx * 3 + x_dim] += -3*prefactor*dx*dz;
                // hessian_out[j_idx * 3 * N * 3 + z_dim * N * 3 + i_idx * 3 + y_dim] += -3*prefactor*dy*dz;
                // hessian_out[j_idx * 3 * N * 3 + z_dim * N * 3 + i_idx * 3 + z_dim] += prefactor*(d2ij - 3*d2z);

            }

            int srcLane = (threadIdx.x + 1) % WARP_SIZE;

            // printf("tid %d srcLane %d\n", threadIdx.x, srcLane);
            x1 = __shfl_sync(0xffffffff, x1, srcLane);
            y1 = __shfl_sync(0xffffffff, y1, srcLane);
            z1 = __shfl_sync(0xffffffff, z1, srcLane);
            q1 = __shfl_sync(0xffffffff, q1, srcLane);

            shfl_grad_dx = __shfl_sync(0xffffffff, shfl_grad_dx, srcLane);
            shfl_grad_dy = __shfl_sync(0xffffffff, shfl_grad_dy, srcLane);
            shfl_grad_dz = __shfl_sync(0xffffffff, shfl_grad_dz, srcLane);

            shfl_hess_xx = __shfl_sync(0xffffffff, shfl_hess_xx, srcLane);
            // shfl_hess_xy = __shfl_sync(0xffffffff, shfl_hess_xy, srcLane);
            // shfl_hess_xz = __shfl_sync(0xffffffff, shfl_hess_xz, srcLane);
            shfl_hess_yx = __shfl_sync(0xffffffff, shfl_hess_yx, srcLane);
            shfl_hess_yy = __shfl_sync(0xffffffff, shfl_hess_yy, srcLane);
            // shfl_hess_yz = __shfl_sync(0xffffffff, shfl_hess_yz, srcLane);
            shfl_hess_zx = __shfl_sync(0xffffffff, shfl_hess_zx, srcLane);
            shfl_hess_zy = __shfl_sync(0xffffffff, shfl_hess_zy, srcLane);
            shfl_hess_zz = __shfl_sync(0xffffffff, shfl_hess_zz, srcLane);


            j_idx += 1;
            // compute hessians.

        }

        j_idx = tile_y_idx*WARP_SIZE + j_idx % WARP_SIZE;

        // wrap-around pythonic mod %
        // int target_idx = j_idx - 1;
        int target_idx = j_idx;
        // target_idx = (WARP_SIZE + (target_idx % WARP_SIZE)) % WARP_SIZE;

        if(target_idx < n_atoms) {

            // printf("tid %d accumulating to target_idx %d with %f\n", threadIdx.x, target_idx, shfl_grad_dx);

            atomicAdd(grad_out + target_idx*3 + 0, shfl_grad_dx);
            atomicAdd(grad_out + target_idx*3 + 1, shfl_grad_dy);
            atomicAdd(grad_out + target_idx*3 + 2, shfl_grad_dz);

            atomicAdd(hessian_out + target_idx*3*N3 + 0 * N3 + target_idx * 3 + 0, shfl_hess_xx);
            // atomicAdd(hessian_out + target_idx*3*N3 + 0 * N3 + target_idx * 3 + 1, shfl_hess_xy);
            // atomicAdd(hessian_out + target_idx*3*N3 + 0 * N3 + target_idx * 3 + 2, shfl_hess_xz);
            atomicAdd(hessian_out + target_idx*3*N3 + 1 * N3 + target_idx * 3 + 0, shfl_hess_yx);
            atomicAdd(hessian_out + target_idx*3*N3 + 1 * N3 + target_idx * 3 + 1, shfl_hess_yy);
            // atomicAdd(hessian_out + target_idx*3*N3 + 1 * N3 + target_idx * 3 + 2, shfl_hess_yz);
            atomicAdd(hessian_out + target_idx*3*N3 + 2 * N3 + target_idx * 3 + 0, shfl_hess_zx);
            atomicAdd(hessian_out + target_idx*3*N3 + 2 * N3 + target_idx * 3 + 1, shfl_hess_zy);
            atomicAdd(hessian_out + target_idx*3*N3 + 2 * N3 + target_idx * 3 + 2, shfl_hess_zz);
        }

    }

    if(i_idx < n_atoms) {


        atomicAdd(grad_out + i_idx*3 + 0, grad_dx);
        atomicAdd(grad_out + i_idx*3 + 1, grad_dy);
        atomicAdd(grad_out + i_idx*3 + 2, grad_dz);

        atomicAdd(hessian_out + i_idx*3*N3 + 0 * N3 + i_idx * 3 + 0, hess_xx);
        // atomicAdd(hessian_out + i_idx*3*N3 + 0 * N3 + i_idx * 3 + 1, hess_xy);
        // atomicAdd(hessian_out + i_idx*3*N3 + 0 * N3 + i_idx * 3 + 2, hess_xz);
        atomicAdd(hessian_out + i_idx*3*N3 + 1 * N3 + i_idx * 3 + 0, hess_yx);
        atomicAdd(hessian_out + i_idx*3*N3 + 1 * N3 + i_idx * 3 + 1, hess_yy);
        // atomicAdd(hessian_out + i_idx*3*N3 + 1 * N3 + i_idx * 3 + 2, hess_yz);
        atomicAdd(hessian_out + i_idx*3*N3 + 2 * N3 + i_idx * 3 + 0, hess_zx);
        atomicAdd(hessian_out + i_idx*3*N3 + 2 * N3 + i_idx * 3 + 1, hess_zy);
        atomicAdd(hessian_out + i_idx*3*N3 + 2 * N3 + i_idx * 3 + 2, hess_zz);


    }
    // for(int j_idx=i_idx+1; j_idx < n_atoms; j_idx++) {

        // NumericType x1 = coords[j_idx*3+0];
        // NumericType y1 = coords[j_idx*3+1];
        // NumericType z1 = coords[j_idx*3+2];

        // avoid secondary load from global ram by shuffling instead
        // NumericType x1 = coords[0*n_atoms+j_idx];
        // NumericType y1 = coords[1*n_atoms+j_idx];
        // NumericType z1 = coords[2*n_atoms+j_idx];

        // NumericType q1 = params[param_idxs[j_idx]];
        // // NumericType q1 = -0.5;

        // NumericType dx = x0 - x1;
        // NumericType dy = y0 - y1;
        // NumericType dz = z0 - z1;
        // NumericType d2x = dx*dx;
        // NumericType d2y = dy*dy;
        // NumericType d2z = dz*dz;

        // NumericType d2ij = d2x + d2y + d2z;
        // NumericType dij = sqrtf(d2ij);
        // NumericType d3ij = dij*d2ij;
        // NumericType d5ij = d3ij*d2ij;

        // NumericType sij = scale_matrix[i_idx*n_atoms + j_idx];

        // NumericType so4eq01 = sij*ONE_4PI_EPS0*q0*q1;

        // // atom_nrg += so4eq01/dij;

        // NumericType grad_prefactor = so4eq01/d3ij;
        // NumericType hess_prefactor = so4eq01/d5ij;


        // grad_out[i_idx*3 + 0] += grad_prefactor*(-dx);
        // grad_out[i_idx*3 + 1] += grad_prefactor*(-dy);
        // grad_out[i_idx*3 + 2] += grad_prefactor*(-dz);

        // // optimize to use shared diagonal trick later
        // grad_out[j_idx*3 + 0] += grad_prefactor*dx;
        // grad_out[j_idx*3 + 1] += grad_prefactor*dy;
        // grad_out[j_idx*3 + 2] += grad_prefactor*dz;

        // // const int x_dim = 0;
        // // const int 1 = 1;
        // // const int 2 = 2;

        // int i_block = i_idx * 3 * N3;
        // int j_block = j_idx * 3 * N3;

        // hessian_out[i_block + 0 * N3 + i_idx * 3 + 0] += hess_prefactor*(-d2ij + 3*d2x);
        // hessian_out[i_block + 0 * N3 + i_idx * 3 + 1] += 3*hess_prefactor*dx*dy;
        // hessian_out[i_block + 0 * N3 + i_idx * 3 + 2] += 3*hess_prefactor*dx*dz;

        // hessian_out[i_block + 1 * N3 + i_idx * 3 + 0] += 3*hess_prefactor*dx*dy;
        // hessian_out[i_block + 1 * N3 + i_idx * 3 + 1] += hess_prefactor*(-d2ij + 3*d2y);
        // hessian_out[i_block + 1 * N3 + i_idx * 3 + 2] += 3*hess_prefactor*dy*dz;

        // hessian_out[i_block + 2 * N3 + i_idx * 3 + 0] += 3*hess_prefactor*dx*dz;
        // hessian_out[i_block + 2 * N3 + i_idx * 3 + 1] += 3*hess_prefactor*dy*dz;
        // hessian_out[i_block + 2 * N3 + i_idx * 3 + 2] += hess_prefactor*(-d2ij + 3*d2z);

        // insanely slow - symmetrize and optimize.

        // NumericType a = hess_prefactor*(d2ij - 3*d2x);
        // NumericType b = -3*hess_prefactor*dx*dy;
        // NumericType c = -3*hess_prefactor*dx*dz;
        // NumericType d = -3*hess_prefactor*dx*dy;
        // NumericType e = hess_prefactor*(d2ij - 3*d2y);
        // NumericType f = -3*hess_prefactor*dy*dz;
        // NumericType g = -3*hess_prefactor*dx*dz;
        // NumericType h = -3*hess_prefactor*dy*dz;
        // NumericType i = hess_prefactor*(d2ij - 3*d2z);

        // const int ij0 = i_block + 0 * N3 + j_idx * 3;
        // const int ij1 = i_block + 1 * N3 + j_idx * 3;
        // const int ij2 = i_block + 2 * N3 + j_idx * 3;

        // hessian_out[ij0 + 0] += a;
        // hessian_out[ij0 + 1] += b;
        // hessian_out[ij0 + 2] += c;
        // hessian_out[ij1 + 0] += d;
        // hessian_out[ij1 + 1] += e;
        // hessian_out[ij1 + 2] += f;
        // hessian_out[ij2 + 0] += g;
        // hessian_out[ij2 + 1] += h;
        // hessian_out[ij2 + 2] += i;


        // HESSIAN IS SYMMETRIC - POST-FILL LATer
        // const int ji0 = j_block + 0 * N3 + i_idx * 3;
        // const int ji1 = j_block + 1 * N3 + i_idx * 3;
        // const int ji2 = j_block + 2 * N3 + i_idx * 3;

  //       hessian_out[ji0 + 0] += a;
  //       hessian_out[ji0 + 1] += b;
  //       hessian_out[ji0 + 2] += c;
  //       hessian_out[ji1 + 0] += d;
  //       hessian_out[ji1 + 1] += e;
  //       hessian_out[ji1 + 2] += f;
  //       hessian_out[ji2 + 0] += g;
  //       hessian_out[ji2 + 1] += h;
  //       hessian_out[ji2 + 2] += i;

        // hessian_out[j_block + 0 * N3 + j_idx * 3 + 0] += hess_prefactor*(-d2ij + 3*d2x);
        // hessian_out[j_block + 0 * N3 + j_idx * 3 + 1] += 3*hess_prefactor*dx*dy;
        // hessian_out[j_block + 0 * N3 + j_idx * 3 + 2] += 3*hess_prefactor*dx*dz;
        // hessian_out[j_block + 1 * N3 + j_idx * 3 + 0] += 3*hess_prefactor*dx*dy;
        // hessian_out[j_block + 1 * N3 + j_idx * 3 + 1] += hess_prefactor*(-d2ij + 3*d2y);
        // hessian_out[j_block + 1 * N3 + j_idx * 3 + 2] += 3*hess_prefactor*dy*dz;
        // hessian_out[j_block + 2 * N3 + j_idx * 3 + 0] += 3*hess_prefactor*dx*dz;
        // hessian_out[j_block + 2 * N3 + j_idx * 3 + 1] += 3*hess_prefactor*dy*dz;
        // hessian_out[j_block + 2 * N3 + j_idx * 3 + 2] += hess_prefactor*(-d2ij + 3*d2z);

        // NumericType *mp_out_qj = mp_out + global_param_idxs[param_idxs[j_idx]]*n_atoms*3;

        // NumericType PREPREFACTOR = sij*ONE_4PI_EPS0/d3ij;

        // NumericType PREFACTOR_QI_GRAD = PREPREFACTOR*q1;
        // NumericType PREFACTOR_QJ_GRAD = PREPREFACTOR*q0;

        // a = PREFACTOR_QI_GRAD * dx;
        // b = PREFACTOR_QI_GRAD * dy;
        // c = PREFACTOR_QI_GRAD * dz;

  //       mp_out_qi[i_idx*3 + 0] -= a;
  //       mp_out_qi[i_idx*3 + 1] -= b;
  //       mp_out_qi[i_idx*3 + 2] -= c;
  //       mp_out_qi[j_idx*3 + 0] += a;
  //       mp_out_qi[j_idx*3 + 1] += b;
  //       mp_out_qi[j_idx*3 + 2] += c;

        // a = PREFACTOR_QJ_GRAD * dx;
        // b = PREFACTOR_QJ_GRAD * dy;
        // c = PREFACTOR_QJ_GRAD * dz;

  //       mp_out_qj[i_idx*3 + 0] -= a;
  //       mp_out_qj[i_idx*3 + 1] -= b;
  //       mp_out_qj[i_idx*3 + 2] -= c;
  //       mp_out_qj[j_idx*3 + 0] += a;
  //       mp_out_qj[j_idx*3 + 1] += b;
  //       mp_out_qj[j_idx*3 + 2] += c;

    // }

    // atomicAdd(energy_out, atom_nrg);

};

template <typename NumericType>
ElectrostaticsGPU<NumericType>::ElectrostaticsGPU(
    std::vector<NumericType> params,
    std::vector<size_t> global_param_idxs,
    std::vector<size_t> param_idxs,
    std::vector<NumericType> scale_matrix
) {

    // convert to int version
    std::vector<int> int_global_param_idxs;
    for(auto a : global_param_idxs) {
        int_global_param_idxs.push_back(a);
    }
    std::vector<int> int_param_idxs;
    for(auto a : param_idxs) {
        int_param_idxs.push_back(a);
    }

    gpuErrchk(cudaMalloc((void**)&d_params_, params.size()*sizeof(NumericType)));
    gpuErrchk(cudaMalloc((void**)&d_global_param_idxs_, int_global_param_idxs.size()*sizeof(*d_global_param_idxs_)));
    gpuErrchk(cudaMalloc((void**)&d_param_idxs_, int_param_idxs.size()*sizeof(*d_param_idxs_)));
    gpuErrchk(cudaMalloc((void**)&d_scale_matrix_, scale_matrix.size()*sizeof(NumericType)));

    gpuErrchk(cudaMemcpy(d_params_, &params[0], params.size()*sizeof(NumericType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_global_param_idxs_, &int_global_param_idxs[0], int_global_param_idxs.size()*sizeof(*d_global_param_idxs_), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_param_idxs_, &int_param_idxs[0], int_param_idxs.size()*sizeof(*d_param_idxs_), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_scale_matrix_, &scale_matrix[0], scale_matrix.size()*sizeof(NumericType), cudaMemcpyHostToDevice));

};


template <typename NumericType>
ElectrostaticsGPU<NumericType>::~ElectrostaticsGPU() {

    gpuErrchk(cudaFree(d_params_));
    gpuErrchk(cudaFree(d_global_param_idxs_));
    gpuErrchk(cudaFree(d_param_idxs_));
    gpuErrchk(cudaFree(d_scale_matrix_));

};


template <typename NumericType>
void ElectrostaticsGPU<NumericType>::total_derivative(
    const size_t n_atoms,
    const size_t n_params,
    const NumericType* d_coords, // [N, 3]
    NumericType* d_energy_out, // []
    NumericType* d_grad_out, // [N,3]
    NumericType* d_hessian_out, // [N, 3, N, 3]
    NumericType* d_mp_out // [P, N, 3]
) {

    size_t tpb = 32;
    size_t n_blocks = (n_atoms + tpb - 1) / tpb;

    electrostatics_total_derivative<<<n_blocks, tpb>>>(
        d_coords,
        d_params_, // change to int later?
        d_global_param_idxs_, // change to int later?
        d_param_idxs_,
        d_scale_matrix_,
        d_energy_out,
        d_grad_out,
        d_hessian_out,
        d_mp_out,
        n_params,
        n_atoms);

};


template <typename NumericType>
void ElectrostaticsGPU<NumericType>::total_derivative_cpu(
    const size_t N,
    const size_t P,
    const NumericType* coords, // [N, 3]
    NumericType* energy_out, // []
    NumericType* grad_out, // [N,3]
    NumericType* hessian_out, // [N, 3, N, 3]
    NumericType* mp_out // [P, N, 3]
) {

    NumericType* d_coords; // []
    NumericType* d_energy_out; // []
    NumericType* d_grad_out; // [N,3]
    NumericType* d_hessian_out; // [N, 3, N, 3]
    NumericType* d_mp_out; // [P, N, 3]

    // this is a debugging function.

    gpuErrchk(cudaMalloc((void**)&d_coords, N*3*sizeof(NumericType)));
    gpuErrchk(cudaMalloc((void**)&d_energy_out, sizeof(NumericType)));
    gpuErrchk(cudaMalloc((void**)&d_grad_out, N*3*sizeof(NumericType)));
    gpuErrchk(cudaMalloc((void**)&d_hessian_out, N*3*N*3*sizeof(NumericType)));
    gpuErrchk(cudaMalloc((void**)&d_mp_out, P*N*3*sizeof(NumericType)));

    gpuErrchk(cudaMemcpy(d_coords, coords, N*3*sizeof(NumericType), cudaMemcpyHostToDevice));

    std::cout << "CALLING" << std::endl;
    std::clock_t start; double duration; start = std::clock();

    total_derivative(
        N,
        P,
        d_coords,
        d_energy_out,
        d_grad_out,
        d_hessian_out,
        d_mp_out);

    cudaDeviceSynchronize();

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC; std::cout<<"ES_DURATION: "<< duration <<'\n';

    gpuErrchk(cudaMemcpy(energy_out, d_energy_out, sizeof(NumericType), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(grad_out, d_grad_out, N*3*sizeof(NumericType), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(hessian_out, d_hessian_out, N*3*N*3*sizeof(NumericType), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(mp_out, d_mp_out, P*N*3*sizeof(NumericType), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_coords));
    gpuErrchk(cudaFree(d_energy_out));
    gpuErrchk(cudaFree(d_grad_out));
    gpuErrchk(cudaFree(d_hessian_out));
    gpuErrchk(cudaFree(d_mp_out));

};

}

template class timemachine::ElectrostaticsGPU<float>;
template class timemachine::ElectrostaticsGPU<double>;
