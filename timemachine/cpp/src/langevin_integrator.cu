#include "constants.hpp"
#include "gpu_utils.cuh"
#include "kernel_utils.cuh"
#include "langevin_integrator.hpp"
#include "math_utils.cuh"

#include "kernels/k_integrator.cuh"

namespace timemachine {

LangevinIntegrator::LangevinIntegrator(
    int N, const double *masses, double temperature, double dt, double friction, int seed)
    : N_(N), temperature_(temperature), dt_(dt), friction_(friction), noise_iteration_(0), noise_batch_size_(5),
      runner_() {

    ca_ = exp(-friction * dt);

    const double kT = BOLTZ * temperature;
    const double ccs_adjustment = sqrt(1 - exp(-2 * friction * dt));

    std::vector<double> h_ccs(N_);
    std::vector<double> h_cbs(N_);
    for (int i = 0; i < N_; i++) {
        h_cbs[i] = dt_ / masses[i];
        h_ccs[i] = ccs_adjustment * sqrt(kT / masses[i]);
    }

    d_cbs_ = gpuErrchkCudaMallocAndCopy(h_cbs.data(), N_);
    d_ccs_ = gpuErrchkCudaMallocAndCopy(h_ccs.data(), N_);

    curandErrchk(curandCreateGenerator(&cr_rng_, CURAND_RNG_PSEUDO_DEFAULT));
    cudaSafeMalloc(&d_noise_, round_up_even(noise_batch_size_ * N_ * 3) * sizeof(*d_noise_));
    curandErrchk(curandSetPseudoRandomGeneratorSeed(cr_rng_, seed));

    cudaSafeMalloc(&d_du_dx_, N_ * 3 * sizeof(*d_du_dx_));
}

LangevinIntegrator::~LangevinIntegrator() {
    gpuErrchk(cudaFree(d_cbs_));
    gpuErrchk(cudaFree(d_ccs_));
    gpuErrchk(cudaFree(d_noise_));
    gpuErrchk(cudaFree(d_du_dx_));
    curandErrchk(curandDestroyGenerator(cr_rng_));
}

double LangevinIntegrator::get_temperature() { return this->temperature_; }

void LangevinIntegrator::step_fwd(
    std::vector<std::shared_ptr<BoundPotential>> &bps,
    double *d_x_t,
    double *d_v_t,
    double *d_box_t,
    unsigned int *d_idxs,
    cudaStream_t stream) {
    const int D = 3;

    gpuErrchk(cudaMemsetAsync(d_du_dx_, 0, N_ * D * sizeof(*d_du_dx_), stream));

    runner_.execute_potentials(
        bps,
        N_,
        d_x_t,
        d_box_t,
        d_du_dx_, // we only need the forces
        nullptr,
        nullptr,
        stream);

    noise_iteration_ = noise_iteration_ % noise_batch_size_;
    if (noise_iteration_ == 0) {
        // Generating noise can be expensive, doing it in batches for efficiency
        curandErrchk(curandSetStream(cr_rng_, stream));
        curandErrchk(templateCurandNormal(cr_rng_, d_noise_, round_up_even(noise_batch_size_ * N_ * D), 0.0, 1.0));
    }

    size_t tpb = DEFAULT_THREADS_PER_BLOCK;
    // Run in blocks so as to ensure that neighboring threads are modifying the same atom
    // which allows more efficient memory accesses/writes
    size_t n_blocks = ceil_divide(N_ * D, tpb);

    update_forward_baoab<double><<<n_blocks, tpb, 0, stream>>>(
        N_, D, ca_, d_idxs, d_cbs_, d_ccs_, d_noise_ + (noise_iteration_ * N_ * D), d_x_t, d_v_t, d_du_dx_, 0.5 * dt_);
    gpuErrchk(cudaPeekAtLastError());
    noise_iteration_++;
}

void LangevinIntegrator::initialize(
    std::vector<std::shared_ptr<BoundPotential>> &bps,
    double *d_x_t,
    double *d_v_t,
    double *d_box_t,
    unsigned int *d_idxs,
    cudaStream_t stream){};

void LangevinIntegrator::finalize(
    std::vector<std::shared_ptr<BoundPotential>> &bps,
    double *d_x_t,
    double *d_v_t,
    double *d_box_t,
    unsigned int *d_idxs,
    cudaStream_t stream){};

} // end namespace timemachine
