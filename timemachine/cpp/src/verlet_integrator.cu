
#include "gpu_utils.cuh"
#include "kernel_utils.cuh"
#include "math_utils.cuh"
#include "verlet_integrator.hpp"

#include "kernels/k_integrator.cuh"

namespace timemachine {

VelocityVerletIntegrator::VelocityVerletIntegrator(int N, double dt, const double *h_cbs)
    : N_(N), dt_(dt), initialized_(false) {

    d_cbs_ = gpuErrchkCudaMallocAndCopy(h_cbs, N);
    gpuErrchk(cudaMalloc(&d_du_dx_, N * 3 * sizeof(*d_du_dx_)));
}

VelocityVerletIntegrator::~VelocityVerletIntegrator() {
    gpuErrchk(cudaFree(d_cbs_));
    gpuErrchk(cudaFree(d_du_dx_));
}

void VelocityVerletIntegrator::step_fwd(
    std::vector<BoundPotential *> &bps,
    double lamb,
    double *d_x_t,
    double *d_v_t,
    double *d_box_t,
    unsigned long long *d_du_dl,
    unsigned int *d_idxs,
    cudaStream_t stream) {

    gpuErrchk(cudaMemsetAsync(d_du_dx_, 0, N_ * 3 * sizeof(*d_du_dx_), stream));

    const int D = 3;
    size_t tpb = warp_size;
    size_t n_blocks = ceil_divide(N_, tpb);
    dim3 dimGrid_dx(n_blocks, D);
    for (int i = 0; i < bps.size(); i++) {
        bps[i]->execute_device(
            N_, d_x_t, d_box_t, lamb, d_du_dx_, nullptr, d_du_dl ? d_du_dl : nullptr, nullptr, stream);
    }
    update_forward_velocity_verlet<double>
        <<<dimGrid_dx, tpb, 0, stream>>>(N_, D, d_idxs, d_cbs_, d_x_t, d_v_t, d_du_dx_, dt_);
    gpuErrchk(cudaPeekAtLastError());
}

void VelocityVerletIntegrator::initialize(
    std::vector<BoundPotential *> &bps,
    double lamb,
    double *d_x_t,
    double *d_v_t,
    double *d_box_t,
    unsigned int *d_idxs,
    cudaStream_t stream) {

    if (initialized_) {
        throw std::runtime_error("initialized twice");
    }
    gpuErrchk(cudaMemsetAsync(d_du_dx_, 0, N_ * 3 * sizeof(*d_du_dx_), stream));

    const int D = 3;
    size_t tpb = warp_size;
    size_t n_blocks = ceil_divide(N_, tpb);
    dim3 dimGrid_dx(n_blocks, D);

    for (int i = 0; i < bps.size(); i++) {
        bps[i]->execute_device(
            N_,
            d_x_t,
            d_box_t,
            lamb,
            d_du_dx_, // we only need the forces
            nullptr,
            nullptr,
            nullptr,
            stream);
    }
    half_step_velocity_verlet<double, true>
        <<<dimGrid_dx, tpb, 0, stream>>>(N_, D, d_idxs, d_cbs_, d_x_t, d_v_t, d_du_dx_, dt_);
    gpuErrchk(cudaPeekAtLastError());
    initialized_ = true;
};

void VelocityVerletIntegrator::finalize(
    std::vector<BoundPotential *> &bps,
    double lamb,
    double *d_x_t,
    double *d_v_t,
    double *d_box_t,
    unsigned int *d_idxs,
    cudaStream_t stream) {

    if (!initialized_) {
        throw std::runtime_error("not initialized");
    }
    gpuErrchk(cudaMemsetAsync(d_du_dx_, 0, N_ * 3 * sizeof(*d_du_dx_), stream));

    const int D = 3;
    size_t tpb = warp_size;
    size_t n_blocks = ceil_divide(N_, tpb);
    dim3 dimGrid_dx(n_blocks, D);

    for (int i = 0; i < bps.size(); i++) {
        bps[i]->execute_device(
            N_,
            d_x_t,
            d_box_t,
            lamb,
            d_du_dx_, // we only need the forces
            nullptr,
            nullptr,
            nullptr,
            stream);
    }
    half_step_velocity_verlet<double, false>
        <<<dimGrid_dx, tpb, 0, stream>>>(N_, D, d_idxs, d_cbs_, d_x_t, d_v_t, d_du_dx_, dt_);
    gpuErrchk(cudaPeekAtLastError());
    initialized_ = false;
};

} // end namespace timemachine
