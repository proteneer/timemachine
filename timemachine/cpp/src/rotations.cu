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

template <typename RealType>
void rotate_coordinates_and_translate_mol_host(
    const int N,
    const int batch_size,
    const double *mol_coords,
    const double *box,
    const RealType *quaternion,
    const RealType *translation,
    double *output) {
    DeviceBuffer<double> d_coords(N * 3);
    d_coords.copy_from(mol_coords);

    DeviceBuffer<double> d_box(3 * 3);
    d_box.copy_from(box);

    DeviceBuffer<RealType> d_quaternion(batch_size * 4);
    d_quaternion.copy_from(quaternion);

    DeviceBuffer<RealType> d_translation(batch_size * 3);
    d_translation.copy_from(translation);

    int mol_offsets[2] = {0, N};
    DeviceBuffer<int> d_mol_offsets(2);
    d_mol_offsets.copy_from(mol_offsets);

    DeviceBuffer<int> d_offset(1);
    DeviceBuffer<int> d_samples(batch_size);
    // Set the sample to the first (0 index)
    gpuErrchk(cudaMemset(d_samples.data, 0, d_samples.size()));
    gpuErrchk(cudaMemset(d_offset.data, 0, d_offset.size()));

    DeviceBuffer<double> d_out(d_coords.length * batch_size);

    const int tpb = DEFAULT_THREADS_PER_BLOCK;

    cudaStream_t stream = static_cast<cudaStream_t>(0);

    k_rotate_and_translate_mols<RealType, true><<<ceil_divide(batch_size, tpb), tpb, 0, stream>>>(
        batch_size,
        batch_size,
        d_offset.data,
        d_coords.data,
        d_box.data,
        d_samples.data,
        d_mol_offsets.data,
        d_quaternion.data,
        d_translation.data,
        d_out.data);
    gpuErrchk(cudaStreamSynchronize(stream));
    d_out.copy_to(output);
}

template void rotate_coordinates_host<float>(const int, const int, const double *, const float *, double *);
template void rotate_coordinates_host<double>(const int, const int, const double *, const double *, double *);

template void rotate_coordinates_and_translate_mol_host<float>(
    const int, const int, const double *, const double *, const float *, const float *, double *);
template void rotate_coordinates_and_translate_mol_host<double>(
    const int, const int, const double *, const double *, const double *, const double *, double *);

} // namespace timemachine
