#include "local_md_utils.hpp"
#include "nonbonded_all_pairs.hpp"
#include "nonbonded_interaction_group.hpp"

#include "fanout_summed_potential.hpp"
#include "summed_potential.hpp"

#include "gpu_utils.cuh"
#include <sstream>

namespace timemachine {

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

void set_nonbonded_potential_idxs(
    std::shared_ptr<Potential> pot, const int num_idxs, const unsigned int *d_idxs, const cudaStream_t stream) {
    if (std::shared_ptr<NonbondedAllPairs<float>> nb_pot = std::dynamic_pointer_cast<NonbondedAllPairs<float>>(pot);
        nb_pot) {
        nb_pot->set_atom_idxs_device(num_idxs, d_idxs, stream);
    } else if (std::shared_ptr<NonbondedAllPairs<double>> nb_pot =
                   std::dynamic_pointer_cast<NonbondedAllPairs<double>>(pot);
               nb_pot) {
        nb_pot->set_atom_idxs_device(num_idxs, d_idxs, stream);
    } else {
        throw std::runtime_error("unable to cast potential to NonbondedAllPairs");
    }
}

void set_nonbonded_ixn_potential_idxs(
    std::shared_ptr<Potential> pot,
    const int num_col_idxs,
    const int num_row_idxs,
    unsigned int *d_col_idxs,
    unsigned int *d_row_idxs,
    const cudaStream_t stream) {
    if (std::shared_ptr<NonbondedInteractionGroup<float>> nb_pot =
            std::dynamic_pointer_cast<NonbondedInteractionGroup<float>>(pot);
        nb_pot) {
        nb_pot->set_atom_idxs_device(num_col_idxs, num_row_idxs, d_col_idxs, d_row_idxs, stream);
    } else if (std::shared_ptr<NonbondedInteractionGroup<double>> nb_pot =
                   std::dynamic_pointer_cast<NonbondedInteractionGroup<double>>(pot);
               nb_pot) {
        nb_pot->set_atom_idxs_device(num_col_idxs, num_row_idxs, d_col_idxs, d_row_idxs, stream);
    } else {
        throw std::runtime_error("Unable to cast potential to NonbondedInteractionGroup");
    }
}

std::shared_ptr<BoundPotential>
construct_ixn_group_potential(const int N, std::shared_ptr<Potential> pot, const int P, const double *d_params) {
    std::vector<double> h_params(P);
    gpuErrchk(cudaMemcpy(&h_params[0], d_params, P * sizeof(*d_params), cudaMemcpyDeviceToHost));
    std::vector<int> dummy_idxs{0};
    std::vector<int> shape{P};
    std::optional<std::set<int>> dummy_col_idxs(std::nullopt);

    if (std::shared_ptr<NonbondedAllPairs<float>> nb_pot = std::dynamic_pointer_cast<NonbondedAllPairs<float>>(pot);
        nb_pot) {
        std::shared_ptr<Potential> ixn_group(new NonbondedInteractionGroup<float>(
            N, dummy_idxs, nb_pot->get_beta(), nb_pot->get_cutoff(), dummy_col_idxs));
        return std::shared_ptr<BoundPotential>(new BoundPotential(ixn_group, shape, &h_params[0]));
    } else if (std::shared_ptr<NonbondedAllPairs<double>> nb_pot =
                   std::dynamic_pointer_cast<NonbondedAllPairs<double>>(pot);
               nb_pot) {
        std::shared_ptr<Potential> ixn_group(new NonbondedInteractionGroup<double>(
            N, dummy_idxs, nb_pot->get_beta(), nb_pot->get_cutoff(), dummy_col_idxs));
        return std::shared_ptr<BoundPotential>(new BoundPotential(ixn_group, shape, &h_params[0]));
    } else {
        throw std::runtime_error("unable to cast potential to NonbondedAllPairs");
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

void verify_local_md_parameters(double radius, double k) {
    // Lower bound on radius selected to be 1 Angstrom, to avoid case where no particles
    // are moved. TBD whether or not this is a good lower bound
    const double min_radius = 0.1;
    if (radius < min_radius) {
        throw std::runtime_error("radius must be greater or equal to " + std::to_string(min_radius));
    }
    if (k < 1.0) {
        throw std::runtime_error("k must be at least one");
    }
    // TBD determine a more precise threshold, currently 10x what has been tested
    const double max_k = 1e6;
    if (k > max_k) {
        std::ostringstream oss;
        oss << "k must be less than than " << max_k;
        throw std::runtime_error(oss.str());
    }
}

} // namespace timemachine
