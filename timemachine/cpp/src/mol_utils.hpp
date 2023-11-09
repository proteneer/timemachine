#pragma once
#include <array>
#include <vector>

namespace timemachine {

void verify_group_idxs(const int N, const std::vector<std::vector<int>> &group_idxs);

// verify_mols_contiguous verifies that all of the atoms in the molecules are sequential.
// IE mol 0 is [0, 1, ..., K] and mol 1 is [K + 1, ....] and so on. This is used by water sampling
// and is an acceptable approach as long as the water molecules are all at the start of the system
void verify_mols_contiguous(const std::vector<std::vector<int>> &group_idxs);

// prepare_group_idxs_for_gpu takes a set of group indices and flattens it into three vectors.
// The first is the atom indices, the second is the mol indices and the last is the mol offsets.
// The first two arrays are both the length of the total number of atoms in the group idxs and the offsets
// are of the number of groups + 1.
std::array<std::vector<int>, 3> prepare_group_idxs_for_gpu(const std::vector<std::vector<int>> &group_idxs);

std::vector<int> get_mol_offsets(const std::vector<std::vector<int>> &group_idxs);

} // namespace timemachine
