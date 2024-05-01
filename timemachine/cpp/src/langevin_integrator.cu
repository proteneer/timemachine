#include "constants.hpp"
#include "gpu_utils.cuh"
#include "kernel_utils.cuh"
#include "langevin_integrator.hpp"
#include "math_utils.cuh"

#include "kernels/k_integrator.cuh"

// Number of batches of noise to generate at one time

static int NOISE_BATCH_SIZE = 10;

namespace timemachine {
template <typename RealType>
LangevinIntegrator<RealType>::LangevinIntegrator(
    int N, const double *masses, double temperature, double dt, double friction, int seed)
    : N_(N), temperature_(temperature), dt_(static_cast<RealType>(dt)), friction_(friction), noise_offset_(0),
      runner_() {

    ca_ = static_cast<RealType>(exp(-friction * dt));

    const double kT = BOLTZ * temperature;
    const double ccs_adjustment = sqrt(1 - exp(-2 * friction * dt));

    std::vector<RealType> h_ccs(N_);
    std::vector<RealType> h_cbs(N_);
    for (int i = 0; i < N_; i++) {
        h_cbs[i] = static_cast<RealType>(dt_ / masses[i]);
        h_ccs[i] = static_cast<RealType>(ccs_adjustment * sqrt(kT / masses[i]));
    }

    d_cbs_ = gpuErrchkCudaMallocAndCopy(h_cbs.data(), N_);
    d_ccs_ = gpuErrchkCudaMallocAndCopy(h_ccs.data(), N_);

    curandErrchk(curandCreateGenerator(&cr_rng_, CURAND_RNG_PSEUDO_DEFAULT));
    cudaSafeMalloc(&d_noise_, round_up_even(NOISE_BATCH_SIZE * N_ * 3) * sizeof(*d_noise_));
    curandErrchk(curandSetPseudoRandomGeneratorSeed(cr_rng_, seed));

    cudaSafeMalloc(&d_du_dx_, N_ * 3 * sizeof(*d_du_dx_));

    // Only need to memset the forces to zero once at initialization; k_update_forward_baoab will zero forces during every step
    gpuErrchk(cudaMemset(d_du_dx_, 0, N_ * 3 * sizeof(*d_du_dx_)));
}
template <typename RealType> LangevinIntegrator<RealType>::~LangevinIntegrator() {
    gpuErrchk(cudaFree(d_cbs_));
    gpuErrchk(cudaFree(d_ccs_));
    gpuErrchk(cudaFree(d_noise_));
    gpuErrchk(cudaFree(d_du_dx_));
    curandErrchk(curandDestroyGenerator(cr_rng_));
}

template <typename RealType> double LangevinIntegrator<RealType>::get_temperature() { return this->temperature_; }

template <typename RealType>
void LangevinIntegrator<RealType>::step_fwd(
    std::vector<std::shared_ptr<BoundPotential>> &bps,
    double *d_x_t,
    double *d_v_t,
    double *d_box_t,
    unsigned int *d_idxs,
    cudaStream_t stream) {
    const int D = 3;

    runner_.execute_potentials(
        bps,
        N_,
        d_x_t,
        d_box_t,
        d_du_dx_, // we only need the forces
        nullptr,
        nullptr,
        stream);

    noise_offset_ = noise_offset_ % NOISE_BATCH_SIZE;
    if (noise_offset_ == 0) {
        // Generating noise can be expensive, generate in batches for efficiency
        curandErrchk(curandSetStream(cr_rng_, stream));
        curandErrchk(templateCurandNormal(cr_rng_, d_noise_, round_up_even(NOISE_BATCH_SIZE * N_ * D), 0.0, 1.0));
    }

    size_t tpb = DEFAULT_THREADS_PER_BLOCK;
    size_t n_blocks = ceil_divide(N_, tpb);

    k_update_forward_baoab<RealType, D><<<n_blocks, tpb, 0, stream>>>(
        N_, ca_, d_idxs, d_cbs_, d_ccs_, d_noise_ + (noise_offset_ * N_ * D), d_x_t, d_v_t, d_du_dx_, dt_);
    gpuErrchk(cudaPeekAtLastError());
    noise_offset_++;
}

template <typename RealType>
void LangevinIntegrator<RealType>::initialize(
    std::vector<std::shared_ptr<BoundPotential>> &bps,
    double *d_x_t,
    double *d_v_t,
    double *d_box_t,
    unsigned int *d_idxs,
    cudaStream_t stream){};

template <typename RealType>
void LangevinIntegrator<RealType>::finalize(
    std::vector<std::shared_ptr<BoundPotential>> &bps,
    double *d_x_t,
    double *d_v_t,
    double *d_box_t,
    unsigned int *d_idxs,
    cudaStream_t stream){};

} // end namespace timemachine
