#include <algorithm>
#include <set>

#include "gpu_utils.cuh"
#include "kernels/k_nonbonded.cuh"
#include "kernels/kernel_utils.cuh"
#include "math_utils.cuh"
#include "nonbonded_mol_energy.hpp"

const static int BLOCK_SIZE = 256;       // 256 blocks appears fastest
const static int THREADS_PER_BLOCK = 64; // 64 threads seem to produce the best outcome on Cuda Arch 8.6

namespace timemachine {

void verify_group_idx(const int N, const std::vector<std::vector<int>> &group_idxs) {
    int num_grouped_atoms = 0;
    std::set<int> group_set;
    for (int i = 0; i < group_idxs.size(); i++) {
        std::vector<int> atoms = group_idxs[i];
        const int num_atoms = atoms.size();
        num_grouped_atoms += num_atoms;
        for (int j = 0; j < num_atoms; j++) {
            int idx = atoms[j];
            if (idx < 0 || idx >= N) {
                throw std::runtime_error("Grouped indices must be between 0 and N");
            }
            group_set.insert(idx);
        }
    }
    // Verify that all of the group indices are unique
    if (group_set.size() != num_grouped_atoms) {
        throw std::runtime_error("All grouped indices must be unique");
    }
}

// prepare_group_idxs_for_gpu takes a set of group indices and flattens it into three vectors.
// The first is the atom indices, the second is the mol indices and the last is the mol offsets.
// The first two arrays are both the length of the total number of atoms in the group idxs and the offsets
// are of the number of groups + 1.
std::array<std::vector<int>, 3> prepare_group_idxs_for_gpu(const std::vector<std::vector<int>> &group_idxs) {
    const int num_mols = group_idxs.size();

    int num_grouped_atoms = 0;
    // Get the total number of atoms
    for (int i = 0; i < num_mols; i++) {
        num_grouped_atoms += group_idxs[i].size();
    }

    int offset = 0;
    // setup the mol idxs and the atom idxs
    std::vector<int> mol_offsets(num_mols + 1);
    std::vector<int> mol_idxs(num_grouped_atoms);
    std::vector<int> atom_idxs(num_grouped_atoms);
    for (int i = 0; i < num_mols; i++) {
        std::vector<int> atoms = group_idxs[i];
        // ASSUMES MOLS ARE MADE UP CONTIGIOUS INDICES IE mols[0] = [0, 1, 2], mols[1] = [3, 2], etc
        // IF mols[0] = [5, 7], mols[1] = [0, 8], THIS WON'T WORK.
        // If this is not the case need a complete mapping of atom_idx to mol_idx
        // Sort the atom indices from smallest to largest so that you can know the range of indices in the molecule
        std::sort(atoms.begin(), atoms.end());
        int num_atoms = atoms.size();
        mol_offsets[i] = offset;
        for (int j = 0; j < num_atoms; j++) {
            mol_idxs[offset + j] = i;
            atom_idxs[offset + j] = atoms[j];
        }
        offset += num_atoms;
    }
    mol_offsets[num_mols] = offset;

    return std::array<std::vector<int>, 3>({atom_idxs, mol_idxs, mol_offsets});
}

template <typename RealType>
NonbondedMolEnergyPotential<RealType>::NonbondedMolEnergyPotential(
    const int N, const std::vector<std::vector<int>> target_mols, const double beta, const double cutoff)
    : N_(N), num_target_mols_(target_mols.size()), beta_(static_cast<RealType>(beta)),
      cutoff_squared_(static_cast<RealType>(cutoff * cutoff)) {
    verify_group_idx(N_, target_mols);

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
    dim3 dimGrid(BLOCK_SIZE, static_cast<int>(d_target_atom_idxs_.length), 1);

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
    k_accumulate_atom_energies_to_per_mol_energies<RealType, BLOCK_SIZE>
        <<<ceil_divide(target_mols, tpb), tpb, 0, stream>>>(
            static_cast<int>(d_target_atom_idxs_.length),
            target_mols,
            d_target_mol_idxs_.data,
            d_target_mol_offsets_.data,
            d_atom_energy_buffer_.data,
            d_output_energies);
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
