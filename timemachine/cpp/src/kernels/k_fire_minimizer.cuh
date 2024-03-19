#include "k_fixed_point.cuh"

namespace timemachine {

template <typename RealType>
__global__ void k_fire_shift(
    const int N,
    const RealType *__restrict__ current_dt      // [1]
    double *__restrict__ x_t,                    // [N, 3]
    double *__restrict__ v_t,                    // [N, 3]
    const unsigned long long *__restrict__ du_dx // [N, 3]
) {

    const RealType dt = current_dt[0];
    const RealType dt_sq = dt * dt;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < N) {

        RealType force_x = -FIXED_TO_FLOAT<RealType>(du_dx[idx * 3 + 0]);
        RealType force_y = -FIXED_TO_FLOAT<RealType>(du_dx[idx * 3 + 1]);
        RealType force_z = -FIXED_TO_FLOAT<RealType>(du_dx[idx * 3 + 2]);

        RealType v_x = v_t[idx * 3 + 0];
        RealType v_y = v_t[idx * 3 + 1];
        RealType v_z = v_t[idx * 3 + 2];

        x_t[idx * 3 + 0] += dt * v_x + dt_sq * force_x;
        x_t[idx * 3 + 1] += dt * v_y + dt_sq * force_y;
        x_t[idx * 3 + 2] += dt * v_z + dt_sq * force_z;

        idx += gridDim.x * blockDim.x;
    }
};

} // namespace timemachine
