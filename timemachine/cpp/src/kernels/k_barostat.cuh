#pragma once

#include "k_fixed_point.cuh"

namespace timemachine {

// k_rescale_positions scales the box and the centroids of groups to evaluate a potential
// barostat move
template <typename RealType>
void __global__ k_rescale_positions(
    const int N,                                     // Number of atoms to shift
    double *__restrict__ coords,                     // Coordinates
    const RealType *__restrict__ length_scale,       // [1]
    const double *__restrict__ box,                  // [9]
    double *__restrict__ scaled_box,                 // [9]
    const int *__restrict__ atom_idxs,               // [N]
    const int *__restrict__ mol_idxs,                // [N]
    const int *__restrict__ mol_offsets,             // [N]
    const unsigned long long *__restrict__ centroids // [N*3]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    RealType center_x = box[0 * 3 + 0] * 0.5f;
    RealType center_y = box[1 * 3 + 1] * 0.5f;
    RealType center_z = box[2 * 3 + 2] * 0.5f;

    RealType scale = static_cast<RealType>(length_scale[0]);
    if (idx == 0) {
        scaled_box[0 * 3 + 0] *= scale;
        scaled_box[1 * 3 + 1] *= scale;
        scaled_box[2 * 3 + 2] *= scale;
    }
    while (idx < N) {
        int atom_idx = atom_idxs[idx];
        int mol_idx = mol_idxs[idx];

        RealType num_atoms = static_cast<RealType>(mol_offsets[mol_idx + 1] - mol_offsets[mol_idx]);

        RealType centroid_x = FIXED_TO_FLOAT<RealType>(centroids[mol_idx * 3 + 0]) / num_atoms;
        RealType centroid_y = FIXED_TO_FLOAT<RealType>(centroids[mol_idx * 3 + 1]) / num_atoms;
        RealType centroid_z = FIXED_TO_FLOAT<RealType>(centroids[mol_idx * 3 + 2]) / num_atoms;

        RealType displacement_x = ((centroid_x - center_x) * scale) + center_x - centroid_x;
        RealType displacement_y = ((centroid_y - center_y) * scale) + center_y - centroid_y;
        RealType displacement_z = ((centroid_z - center_z) * scale) + center_z - centroid_z;

        // centroid of the new molecule
        centroid_x += displacement_x;
        centroid_y += displacement_y;
        centroid_z += displacement_z;

        // compute displacement needed to shift centroid back into the scaled homebox
        RealType scaled_box_x = box[0 * 3 + 0] * scale;
        RealType scaled_box_y = box[1 * 3 + 1] * scale;
        RealType scaled_box_z = box[2 * 3 + 2] * scale;

        RealType new_center_x = scaled_box_x * floor(centroid_x / scaled_box_x);
        RealType new_center_y = scaled_box_y * floor(centroid_y / scaled_box_y);
        RealType new_center_z = scaled_box_z * floor(centroid_z / scaled_box_z);

        // final coordinates
        coords[atom_idx * 3 + 0] += displacement_x - new_center_x;
        coords[atom_idx * 3 + 1] += displacement_y - new_center_y;
        coords[atom_idx * 3 + 2] += displacement_z - new_center_z;

        idx += gridDim.x * blockDim.x;
    }
}

// k_find_group_centroids computes the centroids of a group of atoms.
template <typename RealType>
void __global__ k_find_group_centroids(
    const int N,                               // Number of atoms to shift
    const double *__restrict__ coords,         // Coordinates [N * 3]
    const int *__restrict__ atom_idxs,         // [N]
    const int *__restrict__ mol_idxs,          // [N]
    unsigned long long *__restrict__ centroids // [num_molecules * 3]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < N) {
        int atom_idx = atom_idxs[idx];
        int mol_idx = mol_idxs[idx];
        atomicAdd(centroids + mol_idx * 3 + 0, FLOAT_TO_FIXED<RealType>(coords[atom_idx * 3 + 0]));
        atomicAdd(centroids + mol_idx * 3 + 1, FLOAT_TO_FIXED<RealType>(coords[atom_idx * 3 + 1]));
        atomicAdd(centroids + mol_idx * 3 + 2, FLOAT_TO_FIXED<RealType>(coords[atom_idx * 3 + 2]));
        idx += gridDim.x * blockDim.x;
    }
}

// k_setup_barostat_move performs the initialization for a barostat move. It determines what the the proposed
// volume will be and sets up d_length_scale and d_volume_delta for use in k_decide_move.
template <typename RealType>
void __global__ k_setup_barostat_move(
    const bool adaptive,
    const RealType *__restrict__ rand,     // [2], use first value, second value is metropolis condition
    const double *__restrict__ d_box,      // [3*3]
    RealType *__restrict__ d_volume_delta, // [1]
    double *__restrict__ d_volume_scale,   // [1]
    RealType *__restrict__ d_length_scale, // [1]
    RealType *__restrict__ d_volume        // [1]
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 1) {
        return; // Only a single thread needs to perform this operation
    }
    const RealType volume = d_box[0 * 3 + 0] * d_box[1 * 3 + 1] * d_box[2 * 3 + 2];
    const RealType delta_volume = d_volume_scale[0] * 2 * (rand[0] - 0.5);
    const RealType new_volume = volume + delta_volume;
    if (adaptive && *d_volume_scale == 0.0) {
        *d_volume_scale = 0.01 * volume;
    }
    *d_volume = volume;
    *d_volume_delta = delta_volume;
    *d_length_scale = cbrt(new_volume / volume);
}

// k_decide_move handles the metropolis check for whether or not to accept a barostat move that scales
// the box volume. If the move is accepted then the box will be scaled as well as all of the coordinates.
// It also handles the bookkeeping for the acceptance counters.
template <typename RealType>
void __global__ k_decide_move(
    const int N,
    const bool adaptive,
    const int num_molecules,
    const double kt,
    const double pressure,
    const RealType *__restrict__ rand,           // [2] Use second value
    const RealType *__restrict__ d_volume,       // [1]
    const RealType *__restrict__ d_volume_delta, // [1]
    double *__restrict__ d_volume_scale,         // [1]
    const __int128 *__restrict__ d_init_u,       // [1]
    const __int128 *__restrict__ d_final_u,      // [1]
    double *__restrict__ d_box,                  // [3*3]
    const double *__restrict__ d_box_output,     // [3*3]
    double *__restrict__ d_x,                    // [N*3]
    const double *__restrict__ d_x_output,       // [N*3]
    int *__restrict__ num_accepted,              // [1]
    int *__restrict__ num_attempted              // [1]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Don't compute volume from the box. It leads to a race condition since `d_box` is updated in this kernel
    const RealType volume = *d_volume;
    const RealType volume_delta = d_volume_delta[0];
    const RealType new_volume = volume + volume_delta;
    RealType energy_delta = INFINITY;
    if (!fixed_point_overflow(d_final_u[0]) && !fixed_point_overflow(d_init_u[0])) {
        energy_delta = FIXED_ENERGY_TO_FLOAT<RealType>(d_final_u[0] - d_init_u[0]);
    }

    const RealType w = energy_delta + pressure * volume_delta - num_molecules * kt * std::log(new_volume / volume);

    const bool rejected = w > 0 && rand[1] > std::exp(-w / kt);

    while (idx < N) {
        if (idx == 0) {
            if (!rejected) {
                num_accepted[0]++;
            }
            num_attempted[0]++;
            if (adaptive && num_attempted[0] >= 10) {
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

#pragma unroll 3
        for (int i = 0; i < 3; i++) {
            d_x[idx * 3 + i] = d_x_output[idx * 3 + i];
        }
        idx += gridDim.x * blockDim.x;
    }
}

} // namespace timemachine
