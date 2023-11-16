#pragma once
#include "assert.h"
#include "curand_kernel.h"

namespace timemachine {

// k_generate_translations_within_or_outside_a_sphere will either generate
// translations inside of a sphere or outside of a sphere depending on the value
// of targeting_inner[0]. If it is 1 then it will be in the sphere else outside the sphere
template <typename RealType>
void __global__ k_generate_translations_within_or_outside_a_sphere(
    const int num_translations,
    const double *__restrict__ box,
    const RealType *__restrict__ center,     // [3]
    const int *__restrict__ targeting_inner, // [1]
    const RealType radius,
    curandState_t *__restrict__ rand_states,   // [threads_per_block]
    RealType *__restrict__ output_translations // [num_translations, 3]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const RealType center_x = center[0];
    const RealType center_y = center[1];
    const RealType center_z = center[2];

    curandState_t local_state;
    local_state = rand_states[threadIdx.x];

    if (targeting_inner[0] == 1) {
        while (idx < num_translations) {
            // TBD: Whether or not this randomness needs to be double
            // Done this way initially due to issues with ptxas
            RealType x = curand_normal(&local_state);
            RealType y = curand_normal(&local_state);
            RealType z = curand_normal(&local_state);
            RealType rad = curand_uniform(&local_state);

            RealType norm = sqrt((x * x) + (y * y) + (z * z));

            x /= norm;
            y /= norm;
            z /= norm;

            rad = cbrt(rad);

            output_translations[idx * 3 + 0] = (x * rad * radius) + center_x;
            output_translations[idx * 3 + 1] = (y * rad * radius) + center_y;
            output_translations[idx * 3 + 2] = (z * rad * radius) + center_z;
            idx += gridDim.x * blockDim.x;
        }
    } else {
        const RealType square_radius = radius * radius;
        const RealType box_x = box[0 * 3 + 0];
        const RealType box_y = box[1 * 3 + 1];
        const RealType box_z = box[2 * 3 + 2];

        const RealType inv_box_x = 1 / box_x;
        const RealType inv_box_y = 1 / box_y;
        const RealType inv_box_z = 1 / box_z;

        RealType x;
        RealType y;
        RealType z;

        RealType delta_x;
        RealType delta_y;
        RealType delta_z;

        RealType dist;

        const int num_iterations = 1000;

        while (idx < num_translations) {

            int iterations = 0;
            while (iterations < num_iterations) {
                // TBD: Whether or not this randomness needs to be double
                // Done this way initially due to issues with ptxas
                x = curand_uniform(&local_state) * box_x;
                y = curand_uniform(&local_state) * box_y;
                z = curand_uniform(&local_state) * box_z;

                delta_x = x - center_x;
                delta_y = y - center_y;
                delta_z = z - center_z;

                delta_x -= box_x * nearbyint(delta_x * inv_box_x);
                delta_y -= box_y * nearbyint(delta_y * inv_box_y);
                delta_z -= box_z * nearbyint(delta_z * inv_box_z);

                dist = (delta_x * delta_x) + (delta_y * delta_y) + (delta_z * delta_z);
                if (dist >= square_radius) {
                    output_translations[idx * 3 + 0] = x;
                    output_translations[idx * 3 + 1] = y;
                    output_translations[idx * 3 + 2] = z;
                    break;
                }
                iterations++;
            }
            assert(iterations < num_iterations);
            idx += gridDim.x * blockDim.x;
        }
    }

    rand_states[threadIdx.x] = local_state;
}

} // namespace timemachine
