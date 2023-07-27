#include "barostat.hpp"
#include "constants.hpp"
#include "energy_accumulation.hpp"
#include "fixed_point.hpp"
#include "gpu_utils.cuh"
#include "math_utils.cuh"
#include <algorithm>
#include <set>
#include <stdio.h>

#include "kernels/k_barostat.cuh"

static int RANDOM_BATCH_SIZE = 1000; // Number of batches of random values to generate at a time

namespace timemachine {

MonteCarloBarostat::MonteCarloBarostat(
    const int N,
    const double pressure,    // Expected in Bar
    const double temperature, // Kelvin
    const std::vector<std::vector<int>> group_idxs,
    const int interval,
    const std::vector<std::shared_ptr<BoundPotential>> bps,
    const int seed)
    : N_(N), bps_(bps), pressure_(pressure), temperature_(temperature), interval_(interval), seed_(seed),
      group_idxs_(group_idxs), step_(0), num_grouped_atoms_(0), runner_() {

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

    std::set<int> group_set;
    for (int i = 0; i < num_mols; i++) {
        std::vector<int> atoms = group_idxs[i];
        const int num_atoms = atoms.size();
        num_grouped_atoms_ += num_atoms;
        for (int j = 0; j < num_atoms; j++) {
            int idx = atoms[j];
            if (idx < 0 || idx >= N_) {
                throw std::runtime_error("Grouped indices must be between 0 and N");
            }
            group_set.insert(idx);
        }
    }
    // Verify that all of the group indices are unique
    if (group_set.size() != num_grouped_atoms_) {
        throw std::runtime_error("All grouped indices must be unique");
    }

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

    gpuErrchk(cudaMemset(d_volume_scale_, 0, 1 * sizeof(*d_volume_scale_)));

    cudaSafeMalloc(&d_centroids_, num_mols * 3 * sizeof(*d_centroids_));
    cudaSafeMalloc(&d_atom_idxs_, num_grouped_atoms_ * sizeof(*d_atom_idxs_));
    cudaSafeMalloc(&d_mol_idxs_, num_grouped_atoms_ * sizeof(*d_mol_idxs_));
    cudaSafeMalloc(&d_mol_offsets_, (num_mols + 1) * sizeof(*d_mol_offsets_));

    int offset = 0;
    int mol_offsets[num_mols + 1];
    int mol_idxs[num_grouped_atoms_];
    int atom_idxs[num_grouped_atoms_];
    for (int i = 0; i < num_mols; i++) {
        std::vector<int> atoms = group_idxs[i];
        mol_offsets[i] = offset;
        int num_atoms = atoms.size();
        for (int j = 0; j < num_atoms; j++) {
            mol_idxs[offset + j] = i;
            atom_idxs[offset + j] = atoms[j];
        }
        offset += num_atoms;
    }
    mol_offsets[num_mols] = offset;
    gpuErrchk(cudaMemcpy(d_mol_idxs_, mol_idxs, num_grouped_atoms_ * sizeof(*d_mol_idxs_), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_atom_idxs_, atom_idxs, num_grouped_atoms_ * sizeof(*d_atom_idxs_), cudaMemcpyHostToDevice));
    gpuErrchk(
        cudaMemcpy(d_mol_offsets_, mol_offsets, (num_mols + 1) * sizeof(*d_mol_offsets_), cudaMemcpyHostToDevice));

    this->reset_counters();
};

MonteCarloBarostat::~MonteCarloBarostat() {
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

void MonteCarloBarostat::reset_counters() {
    gpuErrchk(cudaMemset(d_num_accepted_, 0, sizeof(*d_num_accepted_)));
    gpuErrchk(cudaMemset(d_num_attempted_, 0, sizeof(*d_num_attempted_)));
}

void MonteCarloBarostat::inplace_move(
    double *d_x,   // [N*3]
    double *d_box, // [3*3]
    cudaStream_t stream) {
    step_++;
    if (step_ % interval_ != 0) {
        return;
    }

    // Get offset into the d_rand_ array
    int random_offset = (((step_ / interval_) * 2) - 2) % (RANDOM_BATCH_SIZE * 2);

    // Generate random values batches then offset on each move
    // Each move requires two random values, the first is used to adjust the scaling of box in k_setup_barostat_move
    // and the second is used to accept or reject in the metropolis hasting check performed in k_decide_move.
    if (random_offset == 0) {
        curandErrchk(curandSetStream(cr_rng_, stream));
        curandErrchk(curandGenerateUniformDouble(cr_rng_, d_rand_, RANDOM_BATCH_SIZE * 2));
    }

    const int num_molecules = group_idxs_.size();
    gpuErrchk(cudaMemsetAsync(d_centroids_, 0, num_molecules * 3 * sizeof(*d_centroids_), stream));

    k_setup_barostat_move<<<1, 1, 0, stream>>>(
        d_rand_ + random_offset, d_box, d_volume_delta_, d_volume_scale_, d_length_scale_);
    gpuErrchk(cudaPeekAtLastError());

    // Create duplicates of the coords/box that we can modify
    gpuErrchk(cudaMemcpyAsync(d_x_after_, d_x, N_ * 3 * sizeof(*d_x), cudaMemcpyDeviceToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(d_box_after_, d_box, 3 * 3 * sizeof(*d_box_after_), cudaMemcpyDeviceToDevice, stream));

    const int tpb = DEFAULT_THREADS_PER_BLOCK;
    const int blocks = ceil_divide(num_grouped_atoms_, tpb);

    k_find_group_centroids<<<blocks, tpb, 0, stream>>>(
        num_grouped_atoms_, d_x_after_, d_atom_idxs_, d_mol_idxs_, d_centroids_);
    gpuErrchk(cudaPeekAtLastError());

    // Scale centroids
    k_rescale_positions<<<blocks, tpb, 0, stream>>>(
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

    k_decide_move<<<ceil_divide(N_, tpb), tpb, 0, stream>>>(
        N_,
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

void MonteCarloBarostat::set_interval(const int interval) {
    if (interval <= 0) {
        throw std::runtime_error("Barostat interval must be greater than 0");
    }
    interval_ = interval;
    // Clear the step, to ensure user can expect that in N steps the barostat will trigger
    step_ = 0;
}

int MonteCarloBarostat::get_interval() { return interval_; }

void MonteCarloBarostat::set_pressure(const double pressure) {
    pressure_ = pressure;
    // Could have equilibrated and be a large number of steps from shifting volume
    // adjustment, ie num attempted = 300 and num accepted = 150
    this->reset_counters();
}

} // namespace timemachine
