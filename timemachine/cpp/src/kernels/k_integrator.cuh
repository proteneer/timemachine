#include "k_fixed_point.cuh"

namespace timemachine {

template <typename RealType, int D>
__global__ void k_update_forward_baoab(
    const int N,
    const RealType ca,
    const unsigned int *__restrict__ idxs,  // N
    const RealType *__restrict__ cbs,       // N
    const RealType *__restrict__ ccs,       // N
    const RealType *__restrict__ noise,     // N x 3
    double *__restrict__ x_t,               // N x 3
    double *__restrict__ v_t,               // N x 3
    unsigned long long *__restrict__ du_dx, // N x 3
    const RealType dt) {
    static_assert(D == 3);

    int kernel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (kernel_idx < N) {
        int atom_idx = (idxs == nullptr ? kernel_idx : idxs[kernel_idx]);

        if (atom_idx < N) {
            // BAOAB (https://arxiv.org/abs/1203.5428), rotated by half a timestep

            // ca assumed to contain exp(-friction * dt)
            // cbs assumed to contain dt / mass
            // ccs assumed to contain sqrt(1 - exp(-2 * friction * dt)) * sqrt(kT / mass)
            RealType atom_cbs = cbs[atom_idx];
            RealType atom_ccs = ccs[atom_idx];

            RealType force_x = -FIXED_TO_FLOAT<RealType>(du_dx[atom_idx * D + 0]);
            RealType force_y = -FIXED_TO_FLOAT<RealType>(du_dx[atom_idx * D + 1]);
            RealType force_z = -FIXED_TO_FLOAT<RealType>(du_dx[atom_idx * D + 2]);

            RealType v_mid_x = v_t[atom_idx * D + 0] + atom_cbs * force_x;
            RealType v_mid_y = v_t[atom_idx * D + 1] + atom_cbs * force_y;
            RealType v_mid_z = v_t[atom_idx * D + 2] + atom_cbs * force_z;

            v_t[atom_idx * D + 0] = ca * v_mid_x + atom_ccs * noise[atom_idx * D + 0];
            v_t[atom_idx * D + 1] = ca * v_mid_y + atom_ccs * noise[atom_idx * D + 1];
            v_t[atom_idx * D + 2] = ca * v_mid_z + atom_ccs * noise[atom_idx * D + 2];

            x_t[atom_idx * D + 0] += static_cast<RealType>(0.5) * dt * (v_mid_x + v_t[atom_idx * D + 0]);
            x_t[atom_idx * D + 1] += static_cast<RealType>(0.5) * dt * (v_mid_y + v_t[atom_idx * D + 1]);
            x_t[atom_idx * D + 2] += static_cast<RealType>(0.5) * dt * (v_mid_z + v_t[atom_idx * D + 2]);

            // Zero out the forces after using them to avoid having to memset the forces later
            du_dx[atom_idx * D + 0] = 0;
            du_dx[atom_idx * D + 1] = 0;
            du_dx[atom_idx * D + 2] = 0;
        } else if (idxs != nullptr && kernel_idx < N) {
            // Zero out the forces after using them to avoid having to memset the forces later
            // Needed to handle local MD where the next round kernel_idx may actually take a part.
            // Requires idxs[kernel_idx] == kernel_idx when idxs[kernel_idx] != N
            du_dx[kernel_idx * D + 0] = 0;
            du_dx[kernel_idx * D + 1] = 0;
            du_dx[kernel_idx * D + 2] = 0;
        }
        kernel_idx += gridDim.x * blockDim.x;
    }
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

} // namespace timemachine
