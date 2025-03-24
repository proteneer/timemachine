#include "local_md_utils.hpp"
#include "nonbonded_all_pairs.hpp"
#include "nonbonded_interaction_group.hpp"

#include "fanout_summed_potential.hpp"
#include "summed_potential.hpp"

#include "gpu_utils.cuh"
#include <sstream>

namespace timemachine {

int copy_nonbonded_potential_idxs(std::shared_ptr<Potential> pot, const int max_idxs, unsigned int *d_output_idxs) {
    if (std::shared_ptr<NonbondedAllPairs<float>> nb_pot = std::dynamic_pointer_cast<NonbondedAllPairs<float>>(pot);
        nb_pot) {
        unsigned int *data = nb_pot->get_atom_idxs_device();
        int K = nb_pot->get_num_atom_idxs();
        if (K > max_idxs) {
            throw std::runtime_error("indices for all pairs greater than allowed");
        }
        gpuErrchk(cudaMemcpy(d_output_idxs, data, K * sizeof(*data), cudaMemcpyDeviceToDevice));
        return K;
    } else if (std::shared_ptr<NonbondedAllPairs<double>> nb_pot =
                   std::dynamic_pointer_cast<NonbondedAllPairs<double>>(pot);
               nb_pot) {
        unsigned int *data = nb_pot->get_atom_idxs_device();
        int K = nb_pot->get_num_atom_idxs();
        if (K > max_idxs) {
            throw std::runtime_error("indices for all pairs greater than allowed");
        }
        gpuErrchk(cudaMemcpy(d_output_idxs, data, K * sizeof(*data), cudaMemcpyDeviceToDevice));
        return K;
    } else {
        throw std::runtime_error("unable to cast potential to NonbondedAllPairs");
    }
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

std::shared_ptr<BoundPotential> construct_ixn_group_potential(
    const int N, std::shared_ptr<Potential> pot, const int P, const double *d_params, double nblist_padding) {
    std::vector<double> h_params(P);
    gpuErrchk(cudaMemcpy(&h_params[0], d_params, P * sizeof(*d_params), cudaMemcpyDeviceToHost));
    std::vector<int> row_dummy_idxs{0};
    std::vector<int> col_dummy_idxs{1};

    if (N < 2) {
        throw std::runtime_error("N must be greater than 1");
    }

    if (std::shared_ptr<NonbondedAllPairs<float>> nb_pot = std::dynamic_pointer_cast<NonbondedAllPairs<float>>(pot);
        nb_pot) {
        std::shared_ptr<Potential> ixn_group(new NonbondedInteractionGroup<float>(
            N, row_dummy_idxs, col_dummy_idxs, nb_pot->get_beta(), nb_pot->get_cutoff(), false, nblist_padding));
        return std::shared_ptr<BoundPotential>(new BoundPotential(ixn_group, h_params));
    } else if (std::shared_ptr<NonbondedAllPairs<double>> nb_pot =
                   std::dynamic_pointer_cast<NonbondedAllPairs<double>>(pot);
               nb_pot) {
        std::shared_ptr<Potential> ixn_group(new NonbondedInteractionGroup<double>(
            N, row_dummy_idxs, col_dummy_idxs, nb_pot->get_beta(), nb_pot->get_cutoff(), false, nblist_padding));
        return std::shared_ptr<BoundPotential>(new BoundPotential(ixn_group, h_params));
    } else {
        throw std::runtime_error("unable to cast potential to NonbondedAllPairs");
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
