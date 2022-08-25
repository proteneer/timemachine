#include "barostat.hpp"
#include "constants.hpp"
#include "fixed_point.hpp"
#include "gpu_utils.cuh"
#include <algorithm>
#include <cub/cub.cuh>
#include <set>
#include <stdio.h>

#include "kernels/k_fixed_point.cuh"

namespace timemachine {

MonteCarloBarostat::MonteCarloBarostat(
    const int N,
    const double pressure,    // Expected in Bar
    const double temperature, // Kelvin
    const std::vector<std::vector<int>> group_idxs,
    const int interval,
    const std::vector<BoundPotential *> bps,
    const int seed)
    : N_(N), bps_(bps), pressure_(pressure), temperature_(temperature), interval_(interval), seed_(seed),
      group_idxs_(group_idxs), step_(0), num_grouped_atoms_(0), d_sum_storage_(nullptr), d_sum_storage_bytes_(0) {

    // Trigger check that interval is valid
    this->set_interval(interval_);

    // lets not have another facepalm moment again...
    if (temperature < 100.0) {
        std::cout << "warning temperature less than 100K" << std::endl;
    }

    if (pressure > 10.0) {
        std::cout << "warning pressure more than 10bar" << std::endl;
    }

    curandErrchk(curandCreateGenerator(&cr_rng_, CURAND_RNG_PSEUDO_DEFAULT));
    gpuErrchk(cudaMalloc(&d_rand_, 2 * sizeof(double)));
    curandErrchk(curandSetPseudoRandomGeneratorSeed(cr_rng_, seed_));

    const int num_mols = group_idxs_.size();

    gpuErrchk(cudaMalloc(&d_x_after_, N_ * 3 * sizeof(*d_x_after_)));
    gpuErrchk(cudaMalloc(&d_box_after_, 3 * 3 * sizeof(*d_box_after_)));
    gpuErrchk(cudaMalloc(&d_u_buffer_, N_ * sizeof(*d_u_buffer_)));
    gpuErrchk(cudaMalloc(&d_u_after_buffer_, N_ * sizeof(*d_u_after_buffer_)));

    gpuErrchk(cudaMalloc(&d_init_u_, 1 * sizeof(*d_init_u_)));
    gpuErrchk(cudaMalloc(&d_final_u_, 1 * sizeof(*d_final_u_)));

    gpuErrchk(cudaMalloc(&d_num_accepted_, 1 * sizeof(*d_num_accepted_)));
    gpuErrchk(cudaMalloc(&d_num_attempted_, 1 * sizeof(*d_num_attempted_)));

    gpuErrchk(cudaMalloc(&d_volume_, 1 * sizeof(*d_volume_)));
    gpuErrchk(cudaMalloc(&d_length_scale_, 1 * sizeof(*d_length_scale_)));
    gpuErrchk(cudaMalloc(&d_volume_scale_, 1 * sizeof(*d_volume_scale_)));
    gpuErrchk(cudaMalloc(&d_volume_delta_, 1 * sizeof(*d_volume_delta_)));

    gpuErrchk(cudaMemset(d_volume_scale_, 0, 1 * sizeof(*d_volume_scale_)));

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

    gpuErrchk(cudaMalloc(&d_centroids_, num_mols * 3 * sizeof(*d_centroids_)));
    gpuErrchk(cudaMalloc(&d_atom_idxs_, num_grouped_atoms_ * sizeof(*d_atom_idxs_)));
    gpuErrchk(cudaMalloc(&d_mol_idxs_, num_grouped_atoms_ * sizeof(*d_mol_idxs_)));
    gpuErrchk(cudaMalloc(&d_mol_offsets_, (num_mols + 1) * sizeof(*d_mol_offsets_)));

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

    // Use a typed nullptr so cub can calculate space needed to reduce
    unsigned long long *d_in_tmp = nullptr;  // dummy
    unsigned long long *d_out_tmp = nullptr; // dummy

    // Compute amount of space to reduce energies
    cub::DeviceReduce::Sum(d_sum_storage_, d_sum_storage_bytes_, d_in_tmp, d_out_tmp, N_);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMalloc(&d_sum_storage_, d_sum_storage_bytes_));
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
    gpuErrchk(cudaFree(d_sum_storage_));
    gpuErrchk(cudaFree(d_volume_));
    gpuErrchk(cudaFree(d_volume_scale_));
    gpuErrchk(cudaFree(d_volume_delta_));
    gpuErrchk(cudaFree(d_num_accepted_));
    gpuErrchk(cudaFree(d_num_attempted_));
    curandErrchk(curandDestroyGenerator(cr_rng_));
};

void __global__ rescale_positions(
    const int N,                                     // Number of atoms to shift
    double *__restrict__ coords,                     // Cordinates
    const double *__restrict__ length_scale,         // [1]
    const double *__restrict__ box,                  // [9]
    double *__restrict__ scaled_box,                 // [9]
    const int *__restrict__ atom_idxs,               // [N]
    const int *__restrict__ mol_idxs,                // [N]
    const int *__restrict__ mol_offsets,             // [N]
    const unsigned long long *__restrict__ centroids // [N*3]
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    const int atom_idx = atom_idxs[idx];
    const int mol_idx = mol_idxs[idx];

    const double center_x = box[0 * 3 + 0] * 0.5;
    const double center_y = box[1 * 3 + 1] * 0.5;
    const double center_z = box[2 * 3 + 2] * 0.5;

    const double num_atoms = static_cast<double>(mol_offsets[mol_idx + 1] - mol_offsets[mol_idx]);

    const double centroid_x = FIXED_TO_FLOAT<double>(centroids[mol_idx * 3 + 0]) / num_atoms;
    const double centroid_y = FIXED_TO_FLOAT<double>(centroids[mol_idx * 3 + 1]) / num_atoms;
    const double centroid_z = FIXED_TO_FLOAT<double>(centroids[mol_idx * 3 + 2]) / num_atoms;

    const double displacement_x = ((centroid_x - center_x) * length_scale[0]) + center_x - centroid_x;
    const double displacement_y = ((centroid_y - center_y) * length_scale[0]) + center_y - centroid_y;
    const double displacement_z = ((centroid_z - center_z) * length_scale[0]) + center_z - centroid_z;

    coords[atom_idx * 3 + 0] += displacement_x;
    coords[atom_idx * 3 + 1] += displacement_y;
    coords[atom_idx * 3 + 2] += displacement_z;
    if (atom_idx == 0) {
        scaled_box[0 * 3 + 0] *= length_scale[0];
        scaled_box[1 * 3 + 1] *= length_scale[0];
        scaled_box[2 * 3 + 2] *= length_scale[0];
    }
}

void __global__ find_group_centroids(
    const int N,                               // Number of atoms to shift
    const double *__restrict__ coords,         // Coordinates
    const int *__restrict__ atom_idxs,         // [N]
    const int *__restrict__ mol_idxs,          // [N]
    unsigned long long *__restrict__ centroids // [num_molecules * 3]
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    const int atom_idx = atom_idxs[idx];
    const int mol_idx = mol_idxs[idx];
    atomicAdd(centroids + mol_idx * 3 + 0, FLOAT_TO_FIXED<double>(coords[atom_idx * 3 + 0]));
    atomicAdd(centroids + mol_idx * 3 + 1, FLOAT_TO_FIXED<double>(coords[atom_idx * 3 + 1]));
    atomicAdd(centroids + mol_idx * 3 + 2, FLOAT_TO_FIXED<double>(coords[atom_idx * 3 + 2]));
}

void __global__ k_setup_barostat_move(
    const double *__restrict__ rand,     // [2], use first value, second value is metropolis condition
    double *__restrict__ d_box,          // [3*3]
    double *__restrict__ d_volume_delta, // [1]
    double *__restrict__ d_volume_scale, // [1]
    double *__restrict__ d_length_scale  // [1]
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 1) {
        return; // Only a single thread needs to perform this operation
    }
    const double volume = d_box[0 * 3 + 0] * d_box[1 * 3 + 1] * d_box[2 * 3 + 2];
    if (d_volume_scale[0] == 0) {
        d_volume_scale[0] = 0.01 * volume;
    }
    const double delta_volume = d_volume_scale[0] * 2 * (rand[0] - 0.5);
    const double new_volume = volume + delta_volume;
    d_volume_delta[0] = delta_volume;
    d_length_scale[0] = cbrt(new_volume / volume);
}

void __global__ k_decide_move(
    const int N,
    const int num_molecules,
    const double kt,
    const double pressure,
    const double *__restrict__ rand, // [2] Use second value
    double *__restrict__ d_volume_delta,
    double *__restrict__ d_volume_scale,
    const unsigned long long *__restrict__ d_init_u,
    const unsigned long long *__restrict__ d_final_u,
    double *__restrict__ d_box,
    const double *__restrict__ d_box_output,
    double *__restrict__ d_x,
    const double *__restrict__ d_x_output,
    int *__restrict__ num_accepted,
    int *__restrict__ num_attempted) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }

    const double volume = d_box[0 * 3 + 0] * d_box[1 * 3 + 1] * d_box[2 * 3 + 2];
    const double new_volume = volume + d_volume_delta[0];
    const double energy_delta = FIXED_TO_FLOAT<double>(d_final_u[0] - d_init_u[0]);
    const double w = energy_delta + pressure * d_volume_delta[0] - num_molecules * kt * std::log(new_volume / volume);

    const bool rejected = w > 0 && rand[1] > std::exp(-w / kt);
    if (idx == 0) {
        if (!rejected) {
            num_accepted[0]++;
        }
        num_attempted[0]++;
        if (num_attempted[0] >= 10) {
            if (num_accepted[0] < 0.25 * num_attempted[0]) {
                d_volume_scale[0] /= 1.1;
                // Reset the counters
                num_attempted[0] = 0;
                num_accepted[0] = 0;
            } else if (num_accepted[0] > 0.75 * num_attempted[0]) {
                d_volume_scale[0] = min(d_volume_scale[0] * 1.1, volume * 0.3);
                // Reset the counters
                num_attempted[0] = 0;
                num_accepted[0] = 0;
            }
        }
    }
    if (rejected) {
        return;
    }
    // If the mc move was accepted copy all of the data into place

    if (idx < 9) {
        d_box[idx] = d_box_output[idx];
    }

#pragma unroll
    for (int i = 0; i < 3; i++) {
        d_x[idx * 3 + i] = d_x_output[idx * 3 + i];
    }
}

void MonteCarloBarostat::reset_counters() {
    gpuErrchk(cudaMemset(d_num_accepted_, 0, sizeof(*d_num_accepted_)));
    gpuErrchk(cudaMemset(d_num_attempted_, 0, sizeof(*d_num_attempted_)));
}

void MonteCarloBarostat::inplace_move(
    double *d_x,   // [N*3]
    double *d_box, // [3*3]
    const double lambda,
    cudaStream_t stream) {
    step_++;
    if (step_ % interval_ != 0) {
        return;
    }

    curandErrchk(curandSetStream(cr_rng_, stream));
    // Generate scaling and metropolis conditions in one pass
    curandErrchk(curandGenerateUniformDouble(cr_rng_, d_rand_, 2));

    gpuErrchk(cudaMemsetAsync(d_init_u_, 0, sizeof(*d_init_u_), stream));
    gpuErrchk(cudaMemsetAsync(d_final_u_, 0, sizeof(*d_final_u_), stream));
    gpuErrchk(cudaMemsetAsync(d_u_buffer_, 0, N_ * sizeof(*d_u_buffer_), stream));
    gpuErrchk(cudaMemsetAsync(d_u_after_buffer_, 0, N_ * sizeof(*d_u_after_buffer_), stream));

    for (int i = 0; i < bps_.size(); i++) {
        bps_[i]->execute_device(N_, d_x, d_box, lambda, nullptr, nullptr, nullptr, d_u_buffer_, stream);
    }

    cub::DeviceReduce::Sum(d_sum_storage_, d_sum_storage_bytes_, d_u_buffer_, d_init_u_, N_, stream);
    gpuErrchk(cudaPeekAtLastError());

    k_setup_barostat_move<<<1, 1, 0, stream>>>(d_rand_, d_box, d_volume_delta_, d_volume_scale_, d_length_scale_);
    gpuErrchk(cudaPeekAtLastError());

    const int num_molecules = group_idxs_.size();
    gpuErrchk(cudaMemsetAsync(d_centroids_, 0, num_molecules * 3 * sizeof(*d_centroids_), stream));

    // Create duplicates of the coords/box that we can modify
    gpuErrchk(cudaMemcpyAsync(d_x_after_, d_x, N_ * 3 * sizeof(*d_x), cudaMemcpyDeviceToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(d_box_after_, d_box, 3 * 3 * sizeof(*d_box_after_), cudaMemcpyDeviceToDevice, stream));

    const int tpb = 32;
    const int blocks = (num_grouped_atoms_ + tpb - 1) / tpb;

    find_group_centroids<<<blocks, tpb, 0, stream>>>(num_grouped_atoms_, d_x, d_atom_idxs_, d_mol_idxs_, d_centroids_);

    gpuErrchk(cudaPeekAtLastError());

    // Scale centroids
    rescale_positions<<<blocks, tpb, 0, stream>>>(
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

    for (int i = 0; i < bps_.size(); i++) {
        bps_[i]->execute_device(
            N_, d_x_after_, d_box_after_, lambda, nullptr, nullptr, nullptr, d_u_after_buffer_, stream);
    }

    cub::DeviceReduce::Sum(d_sum_storage_, d_sum_storage_bytes_, d_u_after_buffer_, d_final_u_, N_, stream);
    gpuErrchk(cudaPeekAtLastError());

    double pressure = pressure_ * AVOGADRO * 1e-25;
    const double kT = BOLTZ * temperature_;

    const int move_blocks = (N_ + tpb - 1) / tpb;

    k_decide_move<<<move_blocks, tpb, 0, stream>>>(
        N_,
        num_molecules,
        kT,
        pressure,
        d_rand_,
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

    gpuErrchk(cudaPeekAtLastError())
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
