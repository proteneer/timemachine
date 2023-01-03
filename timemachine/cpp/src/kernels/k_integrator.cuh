#include "k_fixed_point.cuh"

template <typename RealType>
__global__ void update_forward_baoab(
    const int N,
    const int D,
    const RealType ca,
    const unsigned int *__restrict__ idxs,
    const RealType *__restrict__ cbs,   // N
    const RealType *__restrict__ ccs,   // N
    const RealType *__restrict__ noise, // N x 3
    RealType *__restrict__ x_t,
    RealType *__restrict__ v_t,
    const unsigned long long *__restrict__ du_dx,
    const RealType dt) {

    int kernel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (kernel_idx >= N) {
        return;
    }
    int atom_idx;
    if (idxs) {
        atom_idx = idxs[kernel_idx];
    } else {
        atom_idx = kernel_idx;
    }
    if (atom_idx >= N) {
        return;
    }

    int d_idx = blockIdx.y;
    int local_idx = atom_idx * D + d_idx;

    RealType force = -FIXED_TO_FLOAT<RealType>(du_dx[local_idx]);

    // BAOAB (https://arxiv.org/abs/1203.5428), rotated by half a timestep

    // ca assumed to contain exp(-friction * dt)
    // cbs assumed to contain dt / mass
    // ccs assumed to contain sqrt(1 - exp(-2 * friction * dt)) * sqrt(kT / mass)
    RealType v_mid = v_t[local_idx] + cbs[atom_idx] * force;

    v_t[local_idx] = ca * v_mid + ccs[atom_idx] * noise[local_idx];
    x_t[local_idx] += 0.5 * dt * (v_mid + v_t[local_idx]);
};

template <typename RealType, bool UPDATE_X>
__global__ void half_step_velocity_verlet(
    const int N,
    const int D,
    const unsigned int *__restrict__ idxs,
    const RealType *__restrict__ cbs, // N, dt / mass
    RealType *__restrict__ x_t,
    RealType *__restrict__ v_t,
    const unsigned long long *__restrict__ du_dx,
    const RealType dt) {
    int kernel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (kernel_idx >= N) {
        return;
    }
    int atom_idx;
    if (idxs) {
        atom_idx = idxs[kernel_idx];
    } else {
        atom_idx = kernel_idx;
    }
    if (atom_idx >= N) {
        return;
    }

    int d_idx = blockIdx.y;
    int local_idx = atom_idx * D + d_idx;

    RealType force = FIXED_TO_FLOAT<RealType>(du_dx[local_idx]);

    v_t[local_idx] += (0.5 * cbs[atom_idx]) * force;
    if (UPDATE_X) {
        x_t[local_idx] += dt * v_t[local_idx];
    }
};

template <typename RealType>
__global__ void update_forward_velocity_verlet(
    const int N,
    const int D,
    const unsigned int *__restrict__ idxs,
    const RealType *__restrict__ cbs, // N, dt / mass
    RealType *__restrict__ x_t,
    RealType *__restrict__ v_t,
    const unsigned long long *__restrict__ du_dx,
    const RealType dt) {
    int kernel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (kernel_idx >= N) {
        return;
    }
    int atom_idx;
    if (idxs) {
        atom_idx = idxs[kernel_idx];
    } else {
        atom_idx = kernel_idx;
    }
    if (atom_idx >= N) {
        return;
    }

    int d_idx = blockIdx.y;
    int local_idx = atom_idx * D + d_idx;

    RealType force = FIXED_TO_FLOAT<RealType>(du_dx[local_idx]);

    v_t[local_idx] += cbs[atom_idx] * force;
    x_t[local_idx] += dt * v_t[local_idx];
};
