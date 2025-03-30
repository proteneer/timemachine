#include "mol_utils.hpp"
#include <algorithm>
#include <set>
#include <stdexcept>

namespace timemachine {

void verify_group_idxs(const int N, const std::vector<std::vector<int>> &group_idxs) {
    unsigned int num_grouped_atoms = 0;
    std::set<int> group_set;
    for (unsigned int i = 0; i < group_idxs.size(); i++) {
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

void verify_mols_contiguous(const std::vector<std::vector<int>> &group_idxs) {
    int last_water_end = group_idxs[0][0] - 1;
    for (unsigned int i = 0; i < group_idxs.size(); i++) {
        std::vector<int> atoms = group_idxs[i];
        const int num_atoms = atoms.size();
        // if (atoms[0] != last_water_end + 1) {
        //     throw std::runtime_error("Molecules are not verify_mols_contiguous: mol " + std::to_string(i) + " " + std::to_string(atoms[0]));
        // }
        if (atoms[0] < last_water_end + 1) {
            printf("Atom 0 %d last water + 1 %d\n", atoms[0], last_water_end + 1);
            throw std::runtime_error(
                "Molecules are not made up of monotonically increasing indices: mol " + std::to_string(i));
        }
        std::sort(atoms.begin(), atoms.end());
        int last_atom = atoms[0];
        for (int j = 1; j < num_atoms; j++) {
            if (last_atom + 1 != atoms[j]) {
                throw std::runtime_error("Molecule " + std::to_string(i) + "is not sequential in atom indices");
            }
            last_atom = atoms[j];
        }
        last_water_end = atoms[num_atoms - 1];
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
        // ASSUMES MOLS ARE MADE UP OF CONTIGIOUS INDICES IE mols[0] = [0, 1, 2], mols[1] = [3, 2], etc
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

std::vector<int> get_mol_offsets(const std::vector<std::vector<int>> &group_idxs) {
    return prepare_group_idxs_for_gpu(group_idxs)[2];
}

std::vector<int> get_atom_indices(const std::vector<std::vector<int>> &group_idxs) {
    return prepare_group_idxs_for_gpu(group_idxs)[0];
}

} // namespace timemachine
