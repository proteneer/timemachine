#include <set>
#include <stdexcept>
#include <string>

#include "nonbonded_all_pairs.hpp"
#include "nonbonded_common.cuh"

#include "fanout_summed_potential.hpp"
#include "summed_potential.hpp"

#include "set_utils.hpp"

namespace timemachine {

void verify_atom_idxs(const int N, const std::vector<int> &atom_idxs, const bool allow_empty) {
    if (atom_idxs.size() == 0) {
        if (allow_empty) {
            // No further checks if we allow the indices to be empty
            return;
        }
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

bool is_summed_potential(std::shared_ptr<Potential> pot) {
    if (std::shared_ptr<FanoutSummedPotential> fanned_potential = std::dynamic_pointer_cast<FanoutSummedPotential>(pot);
        fanned_potential != nullptr) {
        return true;
    } else if (std::shared_ptr<SummedPotential> summed_potential = std::dynamic_pointer_cast<SummedPotential>(pot);
               summed_potential != nullptr) {
        return true;
    }
    return false;
}

bool is_nonbonded_all_pairs_potential(std::shared_ptr<Potential> pot) {
    if (std::shared_ptr<NonbondedAllPairs<float>> nb_pot = std::dynamic_pointer_cast<NonbondedAllPairs<float>>(pot);
        nb_pot) {
        return true;
    } else if (std::shared_ptr<NonbondedAllPairs<double>> nb_pot =
                   std::dynamic_pointer_cast<NonbondedAllPairs<double>>(pot);
               nb_pot) {
        return true;
    }
    return false;
}

// get_nonbonded_all_pair_cutoff_with_padding returns the cutoff plus padding. Using these value can be used
// to validate the box dimensions
double get_nonbonded_all_pair_cutoff_with_padding(std::shared_ptr<Potential> pot) {
    if (std::shared_ptr<NonbondedAllPairs<float>> nb_pot = std::dynamic_pointer_cast<NonbondedAllPairs<float>>(pot);
        nb_pot) {
        return nb_pot->get_cutoff() + nb_pot->get_nblist_padding();
    } else if (std::shared_ptr<NonbondedAllPairs<double>> nb_pot =
                   std::dynamic_pointer_cast<NonbondedAllPairs<double>>(pot);
               nb_pot) {
        return nb_pot->get_cutoff() + nb_pot->get_nblist_padding();
    } else {
        throw std::runtime_error("unable to cast potential to NonbondedAllPairs");
    }
}

// Recursively populate nb_pots potentials with the NonbondedAllPairs
void get_nonbonded_all_pair_potentials(
    std::vector<std::shared_ptr<BoundPotential>> input, std::vector<std::shared_ptr<BoundPotential>> &nb_pots) {
    for (auto pot : input) {
        if (std::shared_ptr<FanoutSummedPotential> fanned_potential =
                std::dynamic_pointer_cast<FanoutSummedPotential>(pot->potential);
            fanned_potential != nullptr) {
            std::vector<double> h_params(pot->size());
            if (pot->size() > 0) {
                pot->d_p->copy_to(&h_params[0]);
            }
            std::vector<int> shape{pot->size()};
            std::vector<std::shared_ptr<BoundPotential>> flattened_bps;
            for (auto summed_pot : fanned_potential->get_potentials()) {
                if (is_summed_potential(summed_pot) || is_nonbonded_all_pairs_potential(summed_pot)) {
                    flattened_bps.push_back(
                        std::shared_ptr<BoundPotential>(new BoundPotential(summed_pot, shape, &h_params[0])));
                }
            }
            get_nonbonded_all_pair_potentials(flattened_bps, nb_pots);
            continue;
        } else if (std::shared_ptr<SummedPotential> summed_potential =
                       std::dynamic_pointer_cast<SummedPotential>(pot->potential);
                   summed_potential != nullptr) {
            std::vector<double> h_params(pot->size());
            int i = 0;
            int offset = 0;
            if (pot->size() > 0) {
                pot->d_p->copy_to(&h_params[0]);
            }
            std::vector<int> shape(1);

            std::vector<std::shared_ptr<BoundPotential>> flattened_bps;
            std::vector<int> param_sizes = summed_potential->get_parameter_sizes();
            for (auto summed_pot : summed_potential->get_potentials()) {

                if (is_summed_potential(summed_pot) || is_nonbonded_all_pairs_potential(summed_pot)) {
                    std::vector<double> slice(h_params.begin() + offset, h_params.begin() + offset + param_sizes[i]);
                    shape[0] = param_sizes[i];
                    flattened_bps.push_back(
                        std::shared_ptr<BoundPotential>(new BoundPotential(summed_pot, shape, &slice[0])));
                }

                offset += param_sizes[i];
                i++;
            }
            get_nonbonded_all_pair_potentials(flattened_bps, nb_pots);
            continue;
        }
        if (is_nonbonded_all_pairs_potential(pot->potential)) {
            nb_pots.push_back(pot);
        }
    }
}
} // namespace timemachine
