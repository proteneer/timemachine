#include <set>
#include <stdexcept>
#include <string>

#include "set_utils.hpp"

void verify_atom_idxs(int N, const std::vector<int> &atom_idxs) {
    if (atom_idxs.size() == 0) {
        throw std::runtime_error("indices can't be empty");
    }
    std::set<int> unique_idxs(atom_idxs.begin(), atom_idxs.end());
    if (unique_idxs.size() != atom_idxs.size()) {
        throw std::runtime_error("atom indices must be unique");
    }
    if (*std::max_element(atom_idxs.begin(), atom_idxs.end()) >= N) {
        throw std::runtime_error("index values must be less than N(" + std::to_string(N) + ")");
    }
    if (*std::min_element(atom_idxs.begin(), atom_idxs.end()) < 0) {
        throw std::runtime_error("index values must be greater or equal to zero");
    }
}
