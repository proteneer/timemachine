#pragma once

#include <vector>

#include "device_buffer.hpp"

namespace timemachine {

// NonbondedMolEnergyPotential computes the energies of one set of molecules against another
template <typename RealType> class NonbondedMolEnergyPotential {

private:
    const int N_;
    const int num_target_mols_;
    const RealType beta_;
    const RealType cutoff_squared_;

    DeviceBuffer<int> d_target_atom_idxs_;
    DeviceBuffer<int> d_target_mol_idxs_;
    DeviceBuffer<int> d_target_mol_offsets_;

    // Intermediate buffer for storing the per atom energies
    DeviceBuffer<__int128> d_atom_energy_buffer_;

public:
    NonbondedMolEnergyPotential(
        const int N, const std::vector<std::vector<int>> &target_mols, const double beta, const double cutoff);

    ~NonbondedMolEnergyPotential() {};

    void mol_energies_device(
        const int N,
        const int target_mols,
        const double *d_coords,      // [N, 3]
        const double *d_params,      // [N, PARAMS_PER_ATOM]
        const double *d_box,         // [3, 3]
        __int128 *d_output_energies, // [target_mols]
        cudaStream_t stream);

    std::vector<__int128>
    mol_energies_host(const int N, const int P, const double *h_coords, const double *h_params, const double *h_box);
};

} // namespace timemachine
