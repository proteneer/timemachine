#include "langevin_integrator.hpp"
#include "nonbonded_all_pairs.hpp"
#include "nonbonded_interaction_group.hpp"

#include "fanout_summed_potential.hpp"
#include "summed_potential.hpp"

#include <typeinfo>

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

    if (std::shared_ptr<NonbondedAllPairs<float>> nb_pot = std::dynamic_pointer_cast<NonbondedAllPairs<float>>(pot);
        nb_pot) {
        std::shared_ptr<Potential> ixn_group(
            new NonbondedInteractionGroup<float>(N, dummy_idxs, nb_pot->get_beta(), nb_pot->get_cutoff()));
        return std::shared_ptr<BoundPotential>(new BoundPotential(ixn_group, shape, &h_params[0]));
    } else if (std::shared_ptr<NonbondedAllPairs<double>> nb_pot =
                   std::dynamic_pointer_cast<NonbondedAllPairs<double>>(pot);
               nb_pot) {
        std::shared_ptr<Potential> ixn_group(
            new NonbondedInteractionGroup<double>(N, dummy_idxs, nb_pot->get_beta(), nb_pot->get_cutoff()));
        return std::shared_ptr<BoundPotential>(new BoundPotential(ixn_group, shape, &h_params[0]));
    } else {
        throw std::runtime_error("unable to cast potential to NonbondedAllPairs");
    }
}

// Recursively flatten the potentials. Important to find specific NonbondedAllPairs potentials for multiple_steps_local which
// can be wrapped in FanoutSummedPotential or SummedPotential objects. Creates copies of bound potentials, for simplicity and safety.
void flatten_potentials(
    std::vector<std::shared_ptr<BoundPotential>> input, std::vector<std::shared_ptr<BoundPotential>> &flattened) {
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
                flattened_bps.push_back(
                    std::shared_ptr<BoundPotential>(new BoundPotential(summed_pot, shape, &h_params[0])));
            }
            flatten_potentials(flattened_bps, flattened);
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
                std::vector<double> slice(h_params.begin() + offset, h_params.begin() + offset + param_sizes[i]);
                shape[0] = param_sizes[i];

                flattened_bps.push_back(
                    std::shared_ptr<BoundPotential>(new BoundPotential(summed_pot, shape, &slice[0])));
                offset += param_sizes[i];
                i++;
            }
            flatten_potentials(flattened_bps, flattened);
            continue;
        }
        flattened.push_back(pot);
    }
}

double Context::_get_temperature() {
    if (std::shared_ptr<LangevinIntegrator> langevin = std::dynamic_pointer_cast<LangevinIntegrator>(intg_);
        langevin != nullptr) {
        return langevin->get_temperature();
    } else {
        throw std::runtime_error("integrator must be LangevinIntegrator.");
    }
}
} // namespace timemachine
