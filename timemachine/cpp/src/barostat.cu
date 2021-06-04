#include "barostat.hpp"
#include "gpu_utils.cuh"
#include "fixed_point.hpp"
#include <algorithm>
#include <stdio.h>
#include <set>

#define AVOGADRO 6.0221367e23
#define BOLTZ 0.008314462618

namespace timemachine {

MonteCarloBarostat::MonteCarloBarostat(
    const int N,
    const double pressure,  // Expected in Bar
    const double temperature, // Kelvin
    const std::vector<std::vector<int> > group_idxs,
    const int interval,
    const std::vector<BoundPotential *> bps,
    const int seed) :
    N_(N),
    pressure_(pressure),
    temperature_(temperature),
    interval_(interval),
    bps_(bps),
    group_idxs_(group_idxs),
    num_grouped_atoms_(0),
    volume_scale_(0),
    num_attempted_(0),
    num_accepted_(0),
    seed_(seed),
    step_(0) {

    // lets not have another facepalm moment again...
    if(temperature < 100.0) {
        std::cout << "warning temperature less than 100K" << std::endl;
    }

    if(pressure > 10.0) {
        std::cout << "warning pressure more than 10bar" << std::endl;
    }

    mt_ = std::mt19937(seed_);
    dist_ = std::uniform_real_distribution<double>(0.0, 1.0);

    //gpuErrchk(cudaMalloc(&d_group_idxs_, group_idxs_.size()*sizeof(*d_group_idxs_)));

    const int num_mols = group_idxs_.size();

    gpuErrchk(cudaMalloc(&d_x_after_, N_*3*sizeof(*d_x_after_)));
    gpuErrchk(cudaMalloc(&d_box_after_, 3*3*sizeof(*d_box_after_)));

    gpuErrchk(cudaMalloc(&d_u_before_, N_*sizeof(*d_u_before_)));
    gpuErrchk(cudaMalloc(&d_u_after_, N_*sizeof(*d_u_after_)));


    std::set<int> group_set;
    for (int i = 0; i < num_mols; i++) {
        std::vector<int> atoms = group_idxs[i];
        const int num_atoms = atoms.size();
        num_grouped_atoms_ += num_atoms;
        for(int j = 0; j < num_atoms; j++) {
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

    gpuErrchk(cudaMalloc(&d_centroids_, num_mols*3*sizeof(*d_centroids_)));
    gpuErrchk(cudaMalloc(&d_atom_idxs_, num_grouped_atoms_*sizeof(*d_atom_idxs_)));
    gpuErrchk(cudaMalloc(&d_mol_idxs_, num_grouped_atoms_*sizeof(*d_mol_idxs_)));
    gpuErrchk(cudaMalloc(&d_mol_offsets_, (num_mols+1)*sizeof(*d_mol_offsets_)));

    int offset = 0;
    int mol_offsets[num_mols+1];
    int mol_idxs[num_grouped_atoms_];
    int atom_idxs[num_grouped_atoms_];
    for (int i = 0; i < num_mols; i++) {
        std::vector<int> atoms = group_idxs[i];
        mol_offsets[i] = offset;
        int num_atoms = atoms.size();
        for (int j = 0; j < num_atoms; j++) {
            mol_idxs[offset+j] = i;
            atom_idxs[offset+j] = atoms[j];
        }
        offset += num_atoms;
    }
    mol_offsets[num_mols] = offset;
    gpuErrchk(cudaMemcpy(d_mol_idxs_, mol_idxs, num_grouped_atoms_*sizeof(*d_mol_idxs_), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_atom_idxs_, atom_idxs, num_grouped_atoms_*sizeof(*d_atom_idxs_), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_mol_offsets_, mol_offsets, (num_mols+1)*sizeof(*d_mol_offsets_), cudaMemcpyHostToDevice));

};

MonteCarloBarostat::~MonteCarloBarostat() {
    gpuErrchk(cudaFree(d_x_after_));
    gpuErrchk(cudaFree(d_centroids_));
    gpuErrchk(cudaFree(d_atom_idxs_));
    gpuErrchk(cudaFree(d_mol_idxs_));
    gpuErrchk(cudaFree(d_mol_offsets_));
    gpuErrchk(cudaFree(d_box_after_));
    gpuErrchk(cudaFree(d_u_before_));
    gpuErrchk(cudaFree(d_u_after_));
};


void __global__ rescale_positions(
    const int N, // Number of atoms to shift
    double * __restrict__ coords, // Cordinates
    const double length_scale,
    const double * __restrict__ box, // [9]
    double * __restrict__ scaled_box, // [9]
    const int * __restrict__ atom_idxs, // [N]
    const int * __restrict__ mol_idxs, // [N]
    const int * __restrict__ mol_offsets, // [N]
    const double * __restrict__ centroids // [N*3]
) {
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    const int atom_idx = atom_idxs[idx];
    const int mol_idx = mol_idxs[idx];

    double center_x = box[0*3+0] * 0.5;
    double center_y = box[1*3+1] * 0.5;
    double center_z = box[2*3+2] * 0.5;

    const double num_atoms = static_cast<double>(mol_offsets[mol_idx+1] - mol_offsets[mol_idx]);

    const double centroid_x = centroids[mol_idx*3+0] / num_atoms;
    const double centroid_y = centroids[mol_idx*3+1] / num_atoms;
    const double centroid_z = centroids[mol_idx*3+2] / num_atoms;

    const double displacement_x = ((centroid_x - center_x) * length_scale) + center_x - centroid_x;
    const double displacement_y = ((centroid_y - center_y) * length_scale) + center_y - centroid_y;
    const double displacement_z = ((centroid_z - center_z) * length_scale) + center_z - centroid_z;

    coords[atom_idx*3+0] += displacement_x;
    coords[atom_idx*3+1] += displacement_y;
    coords[atom_idx*3+2] += displacement_z;
    if (atom_idx == 0) {
        scaled_box[0*3+0] *= length_scale;
        scaled_box[1*3+1] *= length_scale;
        scaled_box[2*3+2] *= length_scale;
    }
}

void __global__ find_group_centroids(
    const int N, // Number of atoms to shift
    const double * __restrict__ coords, // Coordinates
    const int * __restrict__ atom_idxs, // [N]
    const int * __restrict__ mol_idxs, // [N]
    double * __restrict__ centroids // [num_molecules * 3]
) {
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    const int atom_idx = atom_idxs[idx];
    const int mol_idx = mol_idxs[idx];
    atomicAdd(centroids + mol_idx*3+0, coords[atom_idx*3+0]);
    atomicAdd(centroids + mol_idx*3+1, coords[atom_idx*3+2]);
    atomicAdd(centroids + mol_idx*3+2, coords[atom_idx*3+2]);
}

void MonteCarloBarostat::reset_counters() {
    num_attempted_ = 0;
    num_accepted_ = 0;
}

void MonteCarloBarostat::inplace_move(
    double *d_x,
    double *d_box,
    const double lambda
    ) {

    step_++;
    if (step_ % interval_ != 0) {
        return;
    }
    std::vector<double> h_box(9);
    gpuErrchk(cudaMemcpy(&h_box[0], d_box, 9*sizeof(d_box), cudaMemcpyDeviceToHost));

    double volume = h_box[0*3+0]*h_box[1*3+1]*h_box[2*3+2];

    if(volume_scale_ == 0) {
        volume_scale_ = 0.01*volume;
    }

    const int num_molecules = group_idxs_.size();

    const int tpb = 32;
    const int blocks = (num_grouped_atoms_ + tpb - 1) / tpb;

    // Compute the energy of the modified system.

    cudaStream_t stream = static_cast<cudaStream_t>(0);
    gpuErrchk(cudaMemsetAsync(d_u_before_, 0, N_*sizeof(*d_u_before_), stream));
    gpuErrchk(cudaMemsetAsync(d_u_after_, 0, N_*sizeof(*d_u_after_), stream));
    gpuErrchk(cudaMemsetAsync(d_centroids_, 0, num_molecules*3*sizeof(*d_centroids_), stream));

    for(int i=0; i < bps_.size(); i++) {
        bps_[i]->execute_device(
            N_,
            d_x,
            d_box,
            lambda,
            nullptr,
            nullptr,
            nullptr,
            d_u_before_,
            stream // TBD: parallelize me!
        );
    }

    gpuErrchk(cudaMemcpyAsync(d_x_after_, d_x, N_*3*sizeof(*d_x), cudaMemcpyDeviceToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(d_box_after_, d_box, 3*3*sizeof(*d_x), cudaMemcpyDeviceToDevice, stream));

    find_group_centroids<<<blocks, tpb, 0, stream>>>(
        num_grouped_atoms_,
        d_x_after_,
        d_atom_idxs_,
        d_mol_idxs_,
        d_centroids_
    );

    double delta_volume = volume_scale_*2*(dist_(mt_)-0.5);
    double new_volume = volume+delta_volume;
    double length_scale = std::pow(new_volume/volume, 1.0/3.0);

    // Scale centroids
    rescale_positions<<<blocks, tpb, 0, stream>>>(
        num_grouped_atoms_,
        d_x_after_,
        length_scale,
        d_box,
        d_box_after_, // Box will be rescaled by length_scale
        d_atom_idxs_,
        d_mol_idxs_,
        d_mol_offsets_,
        d_centroids_
    );
    gpuErrchk(cudaPeekAtLastError());

    for(int i=0; i < bps_.size(); i++) {
        bps_[i]->execute_device(
            N_,
            d_x_after_,
            d_box_after_,
            lambda,
            nullptr,
            nullptr,
            nullptr,
            d_u_after_,
            stream // TBD: parallelize me!
        );
    }

    double pressure = pressure_*AVOGADRO*1e-25;
    const double kT = BOLTZ*temperature_;

    unsigned long long u_init_agg = 0;
    unsigned long long u_final_agg = 0;

    unsigned long long initial_energy[N_];
    unsigned long long final_energy[N_];

    gpuErrchk(cudaMemcpyAsync(initial_energy, d_u_before_, N_*sizeof(*d_u_before_), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaMemcpyAsync(final_energy, d_u_after_, N_*sizeof(*d_u_after_), cudaMemcpyDeviceToHost, stream));

    gpuErrchk(cudaStreamSynchronize(stream));

    for (int i = 0; i < N_; i++) {
        u_init_agg += initial_energy[i];
        u_final_agg += final_energy[i];
    }

    double u_init = FIXED_TO_FLOAT<double>(u_init_agg);
    double u_final = FIXED_TO_FLOAT<double>(u_final_agg);

    double w = u_final-u_init + pressure*delta_volume - num_molecules*kT*std::log(new_volume/volume);

    if (w > 0 && dist_(mt_) > std::exp(-w/kT)) {
        // Reject the step.
        // Don't modify the coords, keep box the same
        volume = new_volume;
    }
    else {
        num_accepted_++;
        // Replace the coords and box if step accepted
        gpuErrchk(cudaMemcpyAsync(d_x, d_x_after_, N_*3*sizeof(*d_x), cudaMemcpyDeviceToDevice, stream));
        gpuErrchk(cudaMemcpyAsync(d_box, d_box_after_, 3*3*sizeof(*d_box), cudaMemcpyDeviceToDevice, stream));
    }
    num_attempted_++;
    if (num_attempted_ >= 10) {
        if (num_accepted_ < 0.25*num_attempted_) {
            volume_scale_ /= 1.1;
            this->reset_counters();
        }
        else if (num_accepted_ > 0.75*num_attempted_) {
            volume_scale_ = std::min(volume_scale_*1.1, volume*0.3);
            this->reset_counters();
        }
    }
};

void MonteCarloBarostat::set_interval(const int interval){
    interval_ = interval;
    // Clear the step, to ensure user can expect that in N steps the barostat will trigger
    step_ = 0;
}

void MonteCarloBarostat::set_pressure(const double pressure){
    pressure_ = pressure;
    // Could have equilibrated and be a large number of steps from shifting volume
    // adjustment, ie num attempted = 300 and num accepted = 150
    this->reset_counters();
}


}
