#include "barostat.hpp"
#include "constants.hpp"
#include "energy_accumulation.hpp"
#include "fixed_point.hpp"
#include "gpu_utils.cuh"
#include "math_utils.cuh"
#include "mol_utils.hpp"
#include <stdio.h>
#include <variant>

#include "kernels/k_barostat.cuh"

// Number of batches of random values to generate at a time
const static int RANDOM_BATCH_SIZE = 1000;

namespace timemachine {

template <typename RealType>
MonteCarloBarostat<RealType>::MonteCarloBarostat(
    const int N,
    const double pressure,    // Expected in Bar
    const double temperature, // Kelvin
    const std::vector<std::vector<int>> group_idxs,
    const int interval,
    const std::vector<std::shared_ptr<BoundPotential>> bps,
    const int seed,
    const bool adaptive_scaling_enabled,
    const double initial_volume_scale_factor)
    : Mover(interval), N_(N), adaptive_scaling_enabled_(adaptive_scaling_enabled), bps_(bps),
      pressure_(static_cast<RealType>(pressure)), temperature_(static_cast<RealType>(temperature)), seed_(seed),
      group_idxs_(group_idxs), num_grouped_atoms_(0), runner_() {

    // Trigger check that interval is valid
    this->set_interval(interval_);

    // lets not have another facepalm moment again...
    if (temperature < 100.0) {
        std::cout << "warning temperature less than 100K" << std::endl;
    }

    if (pressure > 10.0) {
        std::cout << "warning pressure more than 10bar" << std::endl;
    }

    const int num_mols = group_idxs_.size();

    verify_group_idxs(N, group_idxs);
    // Array of flattened atom indices, mol indices and mol offsets
    std::array<std::vector<int>, 3> flattened_groups = prepare_group_idxs_for_gpu(group_idxs_);

    num_grouped_atoms_ = flattened_groups[0].size();

    curandErrchk(curandCreateGenerator(&cr_rng_, CURAND_RNG_PSEUDO_DEFAULT));
    cudaSafeMalloc(&d_rand_, RANDOM_BATCH_SIZE * 2 * sizeof(*d_rand_));
    curandErrchk(curandSetPseudoRandomGeneratorSeed(cr_rng_, seed_));

    cudaSafeMalloc(&d_x_after_, N_ * 3 * sizeof(*d_x_after_));
    cudaSafeMalloc(&d_box_after_, 3 * 3 * sizeof(*d_box_after_));
    cudaSafeMalloc(&d_u_buffer_, bps_.size() * sizeof(*d_u_buffer_));
    cudaSafeMalloc(&d_u_after_buffer_, bps_.size() * sizeof(*d_u_after_buffer_));

    cudaSafeMalloc(&d_init_u_, 1 * sizeof(*d_init_u_));
    cudaSafeMalloc(&d_final_u_, 1 * sizeof(*d_final_u_));

    cudaSafeMalloc(&d_num_accepted_, 1 * sizeof(*d_num_accepted_));
    cudaSafeMalloc(&d_num_attempted_, 1 * sizeof(*d_num_attempted_));

    cudaSafeMalloc(&d_volume_, 1 * sizeof(*d_volume_));
    cudaSafeMalloc(&d_length_scale_, 1 * sizeof(*d_length_scale_));
    cudaSafeMalloc(&d_volume_scale_, 1 * sizeof(*d_volume_scale_));
    cudaSafeMalloc(&d_volume_delta_, 1 * sizeof(*d_volume_delta_));

    gpuErrchk(cudaMemcpy(
        d_volume_scale_, &initial_volume_scale_factor, 1 * sizeof(*d_volume_scale_), cudaMemcpyHostToDevice));

    cudaSafeMalloc(&d_centroids_, num_mols * 3 * sizeof(*d_centroids_));
    cudaSafeMalloc(&d_mol_offsets_, flattened_groups[2].size() * sizeof(*d_mol_offsets_));

    cudaSafeMalloc(&d_atom_idxs_, num_grouped_atoms_ * sizeof(*d_atom_idxs_));
    cudaSafeMalloc(&d_mol_idxs_, num_grouped_atoms_ * sizeof(*d_mol_idxs_));

    gpuErrchk(cudaMemcpy(
        d_atom_idxs_,
        &flattened_groups[0][0],
        flattened_groups[1].size() * sizeof(*d_atom_idxs_),
        cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(
        d_mol_idxs_,
        &flattened_groups[1][0],
        flattened_groups[0].size() * sizeof(*d_mol_idxs_),
        cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(
        d_mol_offsets_,
        &flattened_groups[2][0],
        flattened_groups[2].size() * sizeof(*d_mol_offsets_),
        cudaMemcpyHostToDevice));

    this->reset_counters();
};

template <typename RealType> MonteCarloBarostat<RealType>::~MonteCarloBarostat() {
    gpuErrchk(cudaFree(d_x_after_));
    gpuErrchk(cudaFree(d_centroids_));
    gpuErrchk(cudaFree(d_atom_idxs_));
    gpuErrchk(cudaFree(d_mol_idxs_));
    gpuErrchk(cudaFree(d_mol_offsets_));
    gpuErrchk(cudaFree(d_box_after_));
    gpuErrchk(cudaFree(d_u_after_buffer_));
    gpuErrchk(cudaFree(d_u_buffer_));
    gpuErrchk(cudaFree(d_init_u_));
    gpuErrchk(cudaFree(d_final_u_));
    gpuErrchk(cudaFree(d_rand_));
    gpuErrchk(cudaFree(d_length_scale_));
    gpuErrchk(cudaFree(d_volume_));
    gpuErrchk(cudaFree(d_volume_scale_));
    gpuErrchk(cudaFree(d_volume_delta_));
    gpuErrchk(cudaFree(d_num_accepted_));
    gpuErrchk(cudaFree(d_num_attempted_));
    curandErrchk(curandDestroyGenerator(cr_rng_));
};

template <typename RealType> void MonteCarloBarostat<RealType>::reset_counters() {
    gpuErrchk(cudaMemset(d_num_accepted_, 0, sizeof(*d_num_accepted_)));
    gpuErrchk(cudaMemset(d_num_attempted_, 0, sizeof(*d_num_attempted_)));
}

template <typename RealType> double MonteCarloBarostat<RealType>::get_volume_scale_factor() {
    double h_scaling;
    gpuErrchk(cudaMemcpy(&h_scaling, d_volume_scale_, 1 * sizeof(*d_volume_scale_), cudaMemcpyDeviceToHost));
    return h_scaling;
}

template <typename RealType>
void MonteCarloBarostat<RealType>::set_volume_scale_factor(const double volume_scale_factor) {
    gpuErrchk(cudaMemcpy(d_volume_scale_, &volume_scale_factor, 1 * sizeof(*d_volume_scale_), cudaMemcpyHostToDevice));
    this->reset_counters();
}

template <typename RealType> bool MonteCarloBarostat<RealType>::get_adaptive_scaling() {
    return this->adaptive_scaling_enabled_;
}

template <typename RealType>
void MonteCarloBarostat<RealType>::set_adaptive_scaling(const bool adaptive_scaling_enabled) {
    this->adaptive_scaling_enabled_ = adaptive_scaling_enabled;
}

template <typename RealType>
void MonteCarloBarostat<RealType>::move(
    const int N,
    double *d_x,   // [N*3]
    double *d_box, // [3*3]
    cudaStream_t stream) {
    if (N != N_) {
        throw std::runtime_error("N != N_");
    }
    this->step_++;
    if (this->step_ % this->interval_ != 0) {
        return;
    }

    // Get offset into the d_rand_ array
    int random_offset = (((step_ / interval_) * 2) - 2) % (RANDOM_BATCH_SIZE * 2);

    // Generate random values batches then offset on each move
    // Each move requires two random values, the first is used to adjust the scaling of box in k_setup_barostat_move
    // and the second is used to accept or reject in the Metropolis-Hastings check performed in k_decide_move.
    if (random_offset == 0) {
        curandErrchk(curandSetStream(cr_rng_, stream));
        curandErrchk(templateCurandUniform(cr_rng_, d_rand_, RANDOM_BATCH_SIZE * 2));
    }

    const int num_molecules = group_idxs_.size();
    gpuErrchk(cudaMemsetAsync(d_centroids_, 0, num_molecules * 3 * sizeof(*d_centroids_), stream));

    k_setup_barostat_move<RealType><<<1, 1, 0, stream>>>(
        adaptive_scaling_enabled_, d_rand_ + random_offset, d_box, d_volume_delta_, d_volume_scale_, d_length_scale_);
    gpuErrchk(cudaPeekAtLastError());

    // Create duplicates of the coords/box that we can modify
    gpuErrchk(cudaMemcpyAsync(d_x_after_, d_x, N_ * 3 * sizeof(*d_x), cudaMemcpyDeviceToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(d_box_after_, d_box, 3 * 3 * sizeof(*d_box_after_), cudaMemcpyDeviceToDevice, stream));

    const int tpb = DEFAULT_THREADS_PER_BLOCK;
    // TBD: For larger systems (20k >) may be better to reduce the number of blocks, rather than
    // matching the number of blocks to be ceil_divide(units_of_work, tpb). The kernels already support this, but
    // at the moment we match the blocks * tpb to equal units_of_work
    const int blocks = ceil_divide(num_grouped_atoms_, tpb);

    k_find_group_centroids<RealType>
        <<<blocks, tpb, 0, stream>>>(num_grouped_atoms_, d_x_after_, d_atom_idxs_, d_mol_idxs_, d_centroids_);
    gpuErrchk(cudaPeekAtLastError());

    // Scale centroids
    k_rescale_positions<RealType><<<blocks, tpb, 0, stream>>>(
        num_grouped_atoms_,
        d_x_after_,
        d_length_scale_,
        d_box,
        d_box_after_, // Box will be rescaled by length_scale
        d_atom_idxs_,
        d_mol_idxs_,
        d_mol_offsets_,
        d_centroids_);
    gpuErrchk(cudaPeekAtLastError());

    runner_.execute_potentials(bps_, N_, d_x, d_box, nullptr, nullptr, d_u_buffer_, stream);
    accumulate_energy(bps_.size(), d_u_buffer_, d_init_u_, stream);

    runner_.execute_potentials(bps_, N_, d_x_after_, d_box_after_, nullptr, nullptr, d_u_after_buffer_, stream);
    accumulate_energy(bps_.size(), d_u_after_buffer_, d_final_u_, stream);

    double pressure = pressure_ * AVOGADRO * 1e-25;
    const double kT = BOLTZ * temperature_;

    k_decide_move<RealType><<<ceil_divide(N_, tpb), tpb, 0, stream>>>(
        N_,
        adaptive_scaling_enabled_,
        num_molecules,
        kT,
        pressure,
        d_rand_ + random_offset,
        d_volume_delta_,
        d_volume_scale_,
        d_init_u_,
        d_final_u_,
        d_box,
        d_box_after_,
        d_x,
        d_x_after_,
        d_num_accepted_,
        d_num_attempted_);
    gpuErrchk(cudaPeekAtLastError());
};

template <typename RealType> void MonteCarloBarostat<RealType>::set_pressure(const double pressure) {
    pressure_ = static_cast<RealType>(pressure);
    // Could have equilibrated and be a large number of steps from shifting volume
    // adjustment, ie num attempted = 300 and num accepted = 150
    this->reset_counters();
}

template class MonteCarloBarostat<float>;
template class MonteCarloBarostat<double>;

} // namespace timemachine
