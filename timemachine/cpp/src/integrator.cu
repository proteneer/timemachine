#include <cstdio>

#include "integrator.hpp"
#include "fixed_point.hpp"
#include "gpu_utils.cuh"

namespace timemachine {

int round_up_even(int count) {
    if(count % 2 == 1) {
        return count += 1;
    } else {
        return count;
    }
}

LangevinIntegrator::LangevinIntegrator(
    int N,
    double dt,
    double ca,
    const double *h_cbs,
    const double *h_ccs,
    int seed
) : N_(N), dt_(dt), ca_(ca) {

    d_cbs_ = gpuErrchkCudaMallocAndCopy(h_cbs, N);
    d_ccs_ = gpuErrchkCudaMallocAndCopy(h_ccs, N);

    curandErrchk(curandCreateGenerator(&cr_rng_, CURAND_RNG_PSEUDO_DEFAULT));
    gpuErrchk(cudaMalloc((void**)&d_noise_, round_up_even(N*3)*sizeof(double)));
    curandErrchk(curandSetPseudoRandomGeneratorSeed(cr_rng_, seed));

}

LangevinIntegrator::~LangevinIntegrator() {
    gpuErrchk(cudaFree(d_cbs_));
    gpuErrchk(cudaFree(d_ccs_));
    gpuErrchk(cudaFree(d_noise_));
    curandErrchk(curandDestroyGenerator(cr_rng_));
}


template <typename RealType>
__global__ void update_forward(
    const int N,
    const int D,
    const RealType ca,
    const RealType *cbs, // N x 3, not P x N x 3, but we could just pass in the first index
    const RealType *ccs, // N
    const RealType *noise,
    RealType *x_t,
    RealType *v_t,
    const unsigned long long *du_dx,
    const RealType dt) {

    int atom_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(atom_idx >= N) {
        return;
    }

    int d_idx = blockIdx.y;
    int local_idx = atom_idx*D + d_idx;

    RealType force = FIXED_TO_FLOAT<RealType>(du_dx[local_idx]);

    // BAOAB (https://arxiv.org/abs/1203.5428), rotated by half a timestep

    // ca assumed to contain exp(-friction * dt)
    // cbs assumed to contain dt / mass
    // ccs assumed to contain sqrt(1 - exp(-2 * friction * dt)) * sqrt(kT / mass)
    RealType v_mid = v_t[local_idx] + cbs[atom_idx] * force;

    v_t[local_idx] = ca * v_mid + ccs[atom_idx] * noise[local_idx];
    x_t[local_idx] += 0.5 * dt * (v_mid + v_t[local_idx]);

};


void LangevinIntegrator::step_fwd(
    double *d_x_t,
    double *d_v_t,
    unsigned long long *d_du_dx_t,
    double *d_box_t_,
    cudaStream_t stream) {

    const int D = 3;
    size_t tpb = 32;
    size_t n_blocks = (N_*D + tpb - 1) / tpb;
    dim3 dimGrid_dx(n_blocks, D);

    curandErrchk(templateCurandNormal(cr_rng_, d_noise_, round_up_even(N_*D), 0.0, 1.0));

    update_forward<double><<<dimGrid_dx, tpb, 0, stream>>>(
        N_,
        D,
        ca_,
        d_cbs_,
        d_ccs_,
        d_noise_,
        d_x_t,
        d_v_t,
        d_du_dx_t,
        dt_
    );

    gpuErrchk(cudaPeekAtLastError());

}

} // end namespace timemachine

// template<typename RealType>
// void step_forward(
//     int N,
//     int D,
//     const RealType ca,
//     const RealType *d_coeff_bs,
//     const RealType *d_coeff_cs, // N
//     const RealType *d_noise_buffer,
//     const RealType *d_x_old,
//     const RealType *d_v_old,
//     const unsigned long long *d_dE_dx_old,
//     const RealType dt,
//     RealType *d_x_new,
//     RealType *d_v_new) {

//     size_t tpb = 32;
//     size_t n_blocks = (N*D + tpb - 1) / tpb;
//     dim3 dimGrid_dx(n_blocks, D);

//     update_forward<<<dimGrid_dx, tpb>>>(
//         N,
//         D,
//         ca,
//         d_coeff_bs,
//         d_coeff_cs,
//         d_noise_buffer,
//         d_x_old,
//         d_v_old,
//         d_dE_dx_old,
//         dt,
//         d_x_new,
//         d_v_new
//     );

//     cudaDeviceSynchronize();
//     gpuErrchk(cudaPeekAtLastError());

// };

// template void step_forward<double>(
//     int N,
//     int D,
//     const double ca,
//     const double *d_coeff_bs,
//     const double *d_coeff_cs,
//     const double *d_noise_buffer,
//     const double *d_x_old,
//     const double *d_v_old,
//     const unsigned long long *d_dE_dx_old,
//     const double dt,
//     double *d_x_new,
//     double *d_v_new);

// template void step_forward<float>(
//     int N,
//     int D,
//     const float ca,
//     const float *d_coeff_bs,
//     const float *d_coeff_cs,
//     const float *d_noise_buffer,
//     const float *d_x_old,
//     const float *d_v_old,
//     const unsigned long long *d_dE_dx_old,
//     const float dt,
//     float *d_x_new,
//     float *d_v_new);

// }