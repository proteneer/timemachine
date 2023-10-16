#include "device_buffer.hpp"
#include "gpu_utils.cuh"
#include "kernels/k_rotations.cuh"
#include "kernels/kernel_utils.cuh"
#include "math_utils.cuh"
#include "rotations.hpp"

namespace timemachine {

// rotate_coordinates_host is used to test the rotation kernel
template <typename RealType>
void rotate_coordinates_host(
    const int N, const int n_rotations, const double *coords, const RealType *quaternions, double *output) {
    DeviceBuffer<double> d_coords(N * 3);
    d_coords.copy_from(coords);

    DeviceBuffer<RealType> d_quats(n_rotations * 4);
    d_quats.copy_from(quaternions);

    DeviceBuffer<double> d_out(n_rotations * N * 3);

    const int tpb = DEFAULT_THREADS_PER_BLOCK;

    dim3 dimGrid(ceil_divide(n_rotations, tpb), N, 1);

    cudaStream_t stream = static_cast<cudaStream_t>(0);

    k_rotate_coordinates<RealType>
        <<<dimGrid, tpb, 0, stream>>>(N, n_rotations, d_coords.data, d_quats.data, d_out.data);
    gpuErrchk(cudaStreamSynchronize(stream));
    d_out.copy_to(output);
}

template void rotate_coordinates_host<float>(const int, const int, const double *, const float *, double *);
template void rotate_coordinates_host<double>(const int, const int, const double *, const double *, double *);

} // namespace timemachine
