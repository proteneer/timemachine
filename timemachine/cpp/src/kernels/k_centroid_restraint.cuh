
#include "surreal.cuh"
#include "../fixed_point.hpp"


__device__ const double PI = 3.14159265358979323846;

template<typename RealType>
void __global__ k_centroid_restraint_inference(
    const int N,     // number of bonds
    const double *coords,  // [n, 3]
    const double lambda,
    const int lambda_flag,
    const int lambda_offset,
    const int *group_a_idxs,
    const int *group_b_idxs,
    const int N_A,
    const int N_B,
    const double *masses, // ignore masses for now
    const double kb,
    const double b0,
    unsigned long long *grad_coords,
    double *du_dl,
    double *energy) {

    // (ytz): ultra shitty inefficient algorithm, optimize later
    const auto t_idx = blockDim.x*blockIdx.x + threadIdx.x;

    if(t_idx != 0) {
        return;
    }

    // stage 0               1         0.5          0
    // stage 1               0         3.5          1
    double lambda_full = lambda_flag*lambda + lambda_offset;

    double group_a_ctr[3] = {0};
    for(int d=0; d < 3; d++) {
        for(int i=0; i < N_A; i++) {
            group_a_ctr[d] += coords[group_a_idxs[i]*3+d];
        }
        group_a_ctr[d] /= N_A;
    }
    double group_b_ctr[3] = {0};

    for(int d=0; d < 3; d++) {
        for(int i=0; i < N_B; i++) {
            group_b_ctr[d] += coords[group_b_idxs[i]*3+d];
        }
        group_b_ctr[d] /= N_B;
    }

    double dx = group_a_ctr[0] - group_b_ctr[0];
    double dy = group_a_ctr[1] - group_b_ctr[1];
    double dz = group_a_ctr[2] - group_b_ctr[2];

    double dij = sqrt(dx*dx + dy*dy + dz*dz);

    double nrg = lambda_full*kb*(dij-b0)*(dij-b0);

    atomicAdd(energy, nrg);
    atomicAdd(du_dl, lambda_flag*kb*(dij-b0)*(dij-b0));

    double du_ddij = 2*lambda_full*kb*(dij-b0);

    // grads
    for(int d=0; d < 3; d++) {
        double ddij_dxi = (group_a_ctr[d] - group_b_ctr[d])/dij;

        for(int i=0; i < N_A; i++) {
            double dx = du_ddij*ddij_dxi/N_A;
            atomicAdd(grad_coords + group_a_idxs[i]*3 + d, static_cast<unsigned long long>((long long) (dx*FIXED_EXPONENT)));
        }
        for(int i=0; i < N_B; i++) {
            double dx = -du_ddij*ddij_dxi/N_B;
            atomicAdd(grad_coords + group_b_idxs[i]*3 + d, static_cast<unsigned long long>((long long) (dx*FIXED_EXPONENT)));
        }
    }

}



template<typename RealType>
void __global__ k_centroid_restraint_jvp(
    const int N,     // number of bonds
    const double *coords_primal,  // [n, 3]
    const double *coords_tangent,
    const double lambda_primal,
    const double lambda_tangent,
    const int lambda_flag,
    const int lambda_offset,
    const int *group_a_idxs,
    const int *group_b_idxs,
    const int N_A,
    const int N_B,
    const double *masses, // ignore masses for now
    const double kb,
    const double b0,
    double *grad_coords_primals,
    double *grad_coords_tangents) {

    // (ytz): ultra shitty inefficient algorithm, optimize later
    const auto t_idx = blockDim.x*blockIdx.x + threadIdx.x;

    if(t_idx != 0) {
        return;
    }

    // stage 0               1         0.5          0
    // stage 1               0         3.5          1
    Surreal<double> lambda(lambda_primal, lambda_tangent);
    Surreal<double> lambda_full = lambda_flag*lambda + lambda_offset;


    Surreal<double> group_a_ctr[3];
    for(int d=0; d < 3; d++) {
        group_a_ctr[d].real = 0;
        group_a_ctr[d].imag = 0;
        for(int i=0; i < N_A; i++) {
            group_a_ctr[d].real += coords_primal[group_a_idxs[i]*3+d];
            group_a_ctr[d].imag += coords_tangent[group_a_idxs[i]*3+d];
        }
        group_a_ctr[d] /= N_A;
    }

    Surreal<double> group_b_ctr[3];
    for(int d=0; d < 3; d++) {
        group_b_ctr[d].real = 0;
        group_b_ctr[d].imag = 0;
        for(int i=0; i < N_B; i++) {
            group_b_ctr[d].real += coords_primal[group_b_idxs[i]*3+d];
            group_b_ctr[d].imag += coords_tangent[group_b_idxs[i]*3+d];
        }
        group_b_ctr[d] /= N_B;
    }

    Surreal<double> dx = group_a_ctr[0] - group_b_ctr[0];
    Surreal<double> dy = group_a_ctr[1] - group_b_ctr[1];
    Surreal<double> dz = group_a_ctr[2] - group_b_ctr[2];

    Surreal<double> dij = sqrt(dx*dx + dy*dy + dz*dz);
    Surreal<double> du_ddij = 2*lambda_full*kb*(dij-b0);

    // grads
    for(int d=0; d < 3; d++) {
        Surreal<double> ddij_dxi = (group_a_ctr[d] - group_b_ctr[d])/dij;

        for(int i=0; i < N_A; i++) {
            Surreal<double> dx = du_ddij*ddij_dxi/N_A;
            atomicAdd(grad_coords_primals + group_a_idxs[i]*3 + d, dx.real);
            atomicAdd(grad_coords_tangents + group_a_idxs[i]*3 + d, dx.imag);
        }
        for(int i=0; i < N_B; i++) {
            Surreal<double> dx = -du_ddij*ddij_dxi/N_B;
            atomicAdd(grad_coords_primals + group_b_idxs[i]*3 + d, dx.real);
            atomicAdd(grad_coords_tangents + group_b_idxs[i]*3 + d, dx.imag);
        }
    }

}
