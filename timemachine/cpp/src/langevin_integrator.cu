#include "gpu_utils.cuh"
#include "kernel_utils.cuh"
#include "langevin_integrator.hpp"
#include "math_utils.cuh"

#include "kernels/k_integrator.cuh"

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

void LangevinIntegrator::step_fwd(
    std::vector<BoundPotential *> &bps,
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

    update_forward_baoab<double>
        <<<dimGrid_dx, tpb, 0, stream>>>(N_, D, ca_, d_cbs_, d_ccs_, d_noise_, d_x_t, d_v_t, d_du_dx_, dt_);

    gpuErrchk(cudaPeekAtLastError());
}

void LangevinIntegrator::initialize(
    std::vector<BoundPotential *> &bps,
    double lamb,
    double *d_x_t,
    double *d_v_t,
    double *d_box_t,
    cudaStream_t stream){};

void LangevinIntegrator::finalize(
    std::vector<BoundPotential *> &bps,
    double lamb,
    double *d_x_t,
    double *d_v_t,
    double *d_box_t,
    cudaStream_t stream){};

} // end namespace timemachine
