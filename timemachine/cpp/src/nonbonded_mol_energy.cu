#include "gpu_utils.cuh"
#include "kernels/k_nonbonded.cuh"
#include "kernels/kernel_utils.cuh"
#include "math_utils.cuh"
#include "mol_utils.hpp"
#include "nonbonded_mol_energy.hpp"

const static int BLOCK_SIZE = 256;       // 256 blocks appears fastest
const static int THREADS_PER_BLOCK = 64; // 64 threads seem to produce the best outcome on Cuda Arch 8.6

namespace timemachine {

template <typename RealType>
NonbondedMolEnergyPotential<RealType>::NonbondedMolEnergyPotential(
    const int N, const std::vector<std::vector<int>> &target_mols, const double beta, const double cutoff)
    : N_(N), num_target_mols_(target_mols.size()), beta_(static_cast<RealType>(beta)),
      cutoff_squared_(static_cast<RealType>(cutoff * cutoff)) {
    verify_group_idxs(N_, target_mols);

    if (num_target_mols_ <= 0) {
        throw std::runtime_error("must provide at least one target mol");
    }

    std::array<std::vector<int>, 3> target_flattened_groups = prepare_group_idxs_for_gpu(target_mols);

    d_target_atom_idxs_.realloc(target_flattened_groups[0].size());
    d_target_atom_idxs_.copy_from(&target_flattened_groups[0][0]);

    d_target_mol_idxs_.realloc(target_flattened_groups[1].size());
    d_target_mol_idxs_.copy_from(&target_flattened_groups[1][0]);

    d_target_mol_offsets_.realloc(target_flattened_groups[2].size());
    d_target_mol_offsets_.copy_from(&target_flattened_groups[2][0]);

    // Need per atom energies per block to account for accumulating between different x blocks
    // that have the same y block index
    d_atom_energy_buffer_.realloc(d_target_atom_idxs_.length * BLOCK_SIZE);
};

template <typename RealType>
void NonbondedMolEnergyPotential<RealType>::mol_energies_device(
    const int N,
    const int target_mols,
    const double *d_coords,      // [N, 3]
    const double *d_params,      // [N, PARAMS_PER_ATOM]
    const double *d_box,         // [3, 3]
    __int128 *d_output_energies, // [target_mols]
    cudaStream_t stream) {
    if (N != N_) {
        throw std::runtime_error("N != N_");
    }

    if (target_mols != num_target_mols_) {
        throw std::runtime_error("target_mols != num_target_mols_");
    }

    const int tpb = THREADS_PER_BLOCK;
    // Brute force num_target_atoms X all atoms
    dim3 dimGrid(BLOCK_SIZE, BLOCK_SIZE, 1);

    gpuErrchk(cudaMemsetAsync(d_atom_energy_buffer_.data, 0, d_atom_energy_buffer_.size(), stream));

    // Compute the per atom energies of all of the molecules of interest
    k_compute_nonbonded_target_atom_energies<RealType, tpb><<<dimGrid, tpb, 0, stream>>>(
        N,
        static_cast<int>(d_target_atom_idxs_.length),
        d_target_atom_idxs_.data,
        d_target_mol_idxs_.data,
        d_target_mol_offsets_.data,
        d_coords,
        d_params,
        d_box,
        beta_,
        cutoff_squared_,
        d_atom_energy_buffer_.data);
    gpuErrchk(cudaPeekAtLastError());

    k_accumulate_atom_energies_to_per_mol_energies<RealType, BLOCK_SIZE><<<target_mols, BLOCK_SIZE, 0, stream>>>(
        target_mols, d_target_mol_offsets_.data, d_atom_energy_buffer_.data, d_output_energies);
    gpuErrchk(cudaPeekAtLastError());
}

template <typename RealType>
std::vector<__int128> NonbondedMolEnergyPotential<RealType>::mol_energies_host(
    const int N, const int P, const double *h_coords, const double *h_params, const double *h_box) {
    DeviceBuffer<double> d_coords(N * 3);
    d_coords.copy_from(h_coords);
    DeviceBuffer<double> d_params(P);
    d_params.copy_from(h_params);
    DeviceBuffer<double> d_box(3 * 3);
    d_box.copy_from(h_box);

    DeviceBuffer<__int128> d_output(num_target_mols_);

    std::vector<__int128> h_output(num_target_mols_);

    cudaStream_t stream = static_cast<cudaStream_t>(0);

    this->mol_energies_device(N, num_target_mols_, d_coords.data, d_params.data, d_box.data, d_output.data, stream);

    gpuErrchk(cudaStreamSynchronize(stream));

    d_output.copy_to(&h_output[0]);

    return h_output;
}

template class NonbondedMolEnergyPotential<float>;
template class NonbondedMolEnergyPotential<double>;

} // namespace timemachine
