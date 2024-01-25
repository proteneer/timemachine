#include "all_atom_energies.hpp"
#include "device_buffer.hpp"
#include "kernels/k_nonbonded.cuh"
#include "kernels/kernel_utils.cuh"
#include "math_utils.cuh"

namespace timemachine {

template <typename RealType>
std::vector<RealType> compute_atom_by_atom_energies(
    const int N,
    const std::vector<int> &target_atoms,
    const std::vector<double> &coords,
    const std::vector<double> &params,
    std::vector<double> &box,
    const RealType nb_beta,
    const RealType cutoff) {
    const DeviceBuffer<int> d_target_atoms(target_atoms);
    const DeviceBuffer<double> d_coords(coords);
    const DeviceBuffer<double> d_params(params);
    const DeviceBuffer<double> d_box(box);
    DeviceBuffer<RealType> d_energy_output(N * target_atoms.size());
    RealType cutoff_squared = cutoff * cutoff;

    cudaStream_t stream = static_cast<cudaStream_t>(0);

    const int tpb = DEFAULT_THREADS_PER_BLOCK;
    dim3 dimGrid(ceil_divide(N, tpb), d_target_atoms.length, 1);

    k_atom_by_atom_energies<<<dimGrid, tpb, 0, stream>>>(
        N,
        static_cast<int>(d_target_atoms.length),
        d_target_atoms.data,
        nullptr, // Use the provided coords to compute the energies
        d_coords.data,
        d_params.data,
        d_box.data,
        nb_beta,
        cutoff_squared,
        d_energy_output.data);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaStreamSynchronize(stream));

    std::vector<RealType> energy_out(d_energy_output.length);
    d_energy_output.copy_to(&energy_out[0]);

    return energy_out;
}

template std::vector<float> compute_atom_by_atom_energies<float>(
    const int N,
    const std::vector<int> &target_atoms,
    const std::vector<double> &coords,
    const std::vector<double> &params,
    std::vector<double> &box,
    float nb_beta,
    float cutoff);
template std::vector<double> compute_atom_by_atom_energies<double>(
    const int N,
    const std::vector<int> &target_atoms,
    const std::vector<double> &coords,
    const std::vector<double> &params,
    std::vector<double> &box,
    double nb_beta,
    double cutoff);

} // namespace timemachine
