#include <cstdio>

#include "fixed_point.hpp"
#include "gpu_utils.cuh"
#include "integrator.hpp"
#include "kernel_utils.cuh"
#include "math_utils.cuh"

namespace timemachine {

int round_up_even(int count) {
    if (count % 2 == 1) {
        return count += 1;
    } else {
        return count;
    }
}

LangevinIntegrator::LangevinIntegrator(int N, double dt, double ca, const double *h_cbs, const double *h_ccs, int seed)
    : N_(N), dt_(dt), ca_(ca) {

    d_cbs_ = gpuErrchkCudaMallocAndCopy(h_cbs, N);
    d_ccs_ = gpuErrchkCudaMallocAndCopy(h_ccs, N);

    curandErrchk(curandCreateGenerator(&cr_rng_, CURAND_RNG_PSEUDO_DEFAULT));
    gpuErrchk(cudaMalloc(&d_noise_, round_up_even(N * 3) * sizeof(double)));
    curandErrchk(curandSetPseudoRandomGeneratorSeed(cr_rng_, seed));

    gpuErrchk(cudaMalloc(&d_du_dx_, N * 3 * sizeof(*d_du_dx_)));
}

LangevinIntegrator::~LangevinIntegrator() {
    gpuErrchk(cudaFree(d_cbs_));
    gpuErrchk(cudaFree(d_ccs_));
    gpuErrchk(cudaFree(d_noise_));
    gpuErrchk(cudaFree(d_du_dx_));
    curandErrchk(curandDestroyGenerator(cr_rng_));
}

template <typename RealType>
__global__ void update_forward(
    const int N,
    const int D,
    const RealType ca,
    const RealType *__restrict__ cbs,   // N
    const RealType *__restrict__ ccs,   // N
    const RealType *__restrict__ noise, // N x 3
    RealType *__restrict__ x_t,
    RealType *__restrict__ v_t,
    const unsigned long long *__restrict__ du_dx,
    const RealType dt) {

    int atom_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom_idx >= N) {
        return;
    }

    int d_idx = blockIdx.y;
    int local_idx = atom_idx * D + d_idx;

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
    std::vector<BoundPotential *> bps,
    double lamb,
    double *d_x_t,
    double *d_v_t,
    double *d_box_t,
    unsigned long long *d_du_dl,
    cudaStream_t stream) {

    gpuErrchk(cudaMemsetAsync(d_du_dx_, 0, N_ * 3 * sizeof(*d_du_dx_), stream));

    for (int i = 0; i < bps.size(); i++) {
        bps[i]->execute_device(
            N_,
            d_x_t,
            d_box_t,
            lamb,
            d_du_dx_, // we only need the forces
            nullptr,
            d_du_dl ? d_du_dl : nullptr,
            nullptr,
            stream);
    }

    const int D = 3;
    size_t tpb = warp_size;
    size_t n_blocks = ceil_divide(N_, tpb);
    dim3 dimGrid_dx(n_blocks, D);

    curandErrchk(curandSetStream(cr_rng_, stream));
    curandErrchk(templateCurandNormal(cr_rng_, d_noise_, round_up_even(N_ * D), 0.0, 1.0));

    update_forward<double>
        <<<dimGrid_dx, tpb, 0, stream>>>(N_, D, ca_, d_cbs_, d_ccs_, d_noise_, d_x_t, d_v_t, d_du_dx_, dt_);

    gpuErrchk(cudaPeekAtLastError());
}

} // end namespace timemachine
