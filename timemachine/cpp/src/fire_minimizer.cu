#include "constants.hpp"
#include "fire_minimizer.hpp"
#include "gpu_utils.cuh"
#include "kernel_utils.cuh"
#include "math_utils.cuh"

#include <cub/cub.cuh>

#include "kernels/k_fire_minimizer.cuh"

namespace timemachine {
template <typename RealType>
FireMinimizer<RealType>::FireMinimizerFireMinimizer(
    const int N,
    const int n_min,
    const double dt_start,
    const double dt_max,
    const double f_inc,
    const double f_dec,
    const double alpha_start,
    const double f_alpha)
    : N_(N), n_min_(n_min), thread_blocks_(ceil_divide(N, DEFAULT_THREADS_PER_BLOCK)),
      dt_max_(static_cast<RealType>(dt_max)), f_alpha_(static_cast<RealType>(f_alpha)),
      f_inc_(static_cast<RealType>(f_inc)), f_dec_(static_cast<RealType>(f_dec)), d_dt_(1), d_alpha_(1), d_step_(1),
      d_accepted_(1), d_force_velo_dot_intermediate_(thread_blocks_), d_force_velo_dot_(1), d_du_dx_old_(N * 3),
      d_du_dx_(N * 3), runner_(), d_temp_buffer_(0), d_temp_storage_buffer_(0) {

    // Convert dt to RealType then copy to d_t;

    size_t sum_storage_bytes = 0;
    void *dummy_temp = nullptr;
    gpuErrchk(cub::DeviceReduce::Sum(
        d_temp_storage_buffer_.data,
        sum_storage_bytes,
        d_force_velo_dot_intermediate_.data,
        d_force_velo_dot_.data,
        thread_blocks_));
    temp_storage_bytes_ = sum_storage_bytes;
    d_temp_storage_buffer_.realloc(temp_storage_bytes_);
}
template <typename RealType> FireMinimizer<RealType>::~FireMinimizer() {}

template <typename RealType>
void FireMinimizer<RealType>::step_fwd(
    std::vector<std::shared_ptr<BoundPotential>> &bps,
    double *d_x_t,
    double *d_v_t,
    double *d_box_t,
    unsigned int *d_idxs,
    cudaStream_t stream) {

    const size_t tpb = DEFAULT_THREADS_PER_BLOCK;

    k_fire_shift<RealType><<<thread_blocks_, tpb, 0, stream>>>(N_, d_dt_.data, d_x_t, d_v_t, d_du_dx_.data);
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaMemsetAsync(d_du_dx_.data, 0, d_du_dx_.size(), stream));

    runner_.execute_potentials(
        bps,
        N_,
        d_x_t,
        d_box_t,
        d_du_dx_.data, // we only need the forces
        nullptr,
        nullptr,
        stream);

    // Update velocities

    // Probably want to do a segment sum on the three parts all at once
    gpuErrchk(cub::DeviceReduce::Sum(
        d_temp_storage_buffer_.data,
        sum_storage_bytes,
        d_force_velo_dot_intermediate_.data,
        d_force_velo_dot_.data,
        thread_blocks_));

    // Update the params
}

template <typename RealType>
void FireMinimizer<RealType>::initialize(
    std::vector<std::shared_ptr<BoundPotential>> &bps,
    double *d_x_t,
    double *d_v_t,
    double *d_box_t,
    unsigned int *d_idxs,
    cudaStream_t stream) {

    gpuErrchk(cudaMemsetAsync(d_du_dx_old_.data, 0, d_du_dx_old_.size(), stream));

    runner_.execute_potentials(
        bps,
        N_,
        d_x_t,
        d_box_t,
        d_du_dx_old_.data, // we only need the forces
        nullptr,
        nullptr,
        stream);
};

template <typename RealType>
void FireMinimizer<RealType>::finalize(
    std::vector<std::shared_ptr<BoundPotential>> &bps,
    double *d_x_t,
    double *d_v_t,
    double *d_box_t,
    unsigned int *d_idxs,
    cudaStream_t stream){};

} // end namespace timemachine
