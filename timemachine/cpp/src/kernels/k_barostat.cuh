#pragma once

#include "k_fixed_point.cuh"

void __global__ rescale_positions(
    const int N,                                     // Number of atoms to shift
    double *__restrict__ coords,                     // Coordinates
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
    if (idx == 0) {
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
    const __int128 *__restrict__ d_init_u,
    const __int128 *__restrict__ d_final_u,
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
    double energy_delta = INFINITY;
    if (!fixed_point_overflow(d_final_u[0]) && !fixed_point_overflow(d_init_u[0])) {
        energy_delta = FIXED_ENERGY_TO_FLOAT<double>(d_final_u[0] - d_init_u[0]);
    }

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
