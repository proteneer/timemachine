#pragma once

#include "device_buffer.hpp"
#include "logsumexp.hpp"
#include "nonbonded_mol_energy.hpp"
#include "weighted_random_sampler.hpp"
#include <array>
#include <vector>

namespace timemachine {

// BDExchangeMove uses biased deletion to move waters randomly in a box. The reference implementation
// is in timemachine/md/exchange/exchange_mover.py::BDExchangeMove
template <typename RealType> class BDExchangeMove {

protected:
    const int N_;
    // Number of atom in all mols
    // All molecules are currently expected to have same number of atoms (typically 3 for waters)
    // Done to avoid having to determine the size of the sample and allows us to test ion sampling by having
    // two different BDExchangemoves
    const int mol_size_;
    const int proposals_per_move_;
    const int num_target_mols_;
    const RealType nb_beta_;
    const RealType beta_; // 1 / kT
    const RealType cutoff_squared_;
    size_t num_attempted_;
    NonbondedMolEnergyPotential<RealType> mol_potential_;
    WeightedRandomSampler<RealType> sampler_;
    LogSumExp<RealType> logsumexp_;
    // Buffer for evaluating moves without touching the original coords
    DeviceBuffer<double> d_intermediate_coords_;
    DeviceBuffer<double> d_params_;
    DeviceBuffer<__int128> d_mol_energy_buffer_;
    DeviceBuffer<RealType> d_sample_per_atom_energy_buffer_; // [mol_size_ * N]
    DeviceBuffer<int> d_atom_idxs_;
    DeviceBuffer<int> d_mol_offsets_;
    DeviceBuffer<RealType> d_log_weights_before_;
    DeviceBuffer<RealType> d_log_weights_after_;
    DeviceBuffer<RealType> d_log_sum_exp_before_; // [2]
    DeviceBuffer<RealType> d_log_sum_exp_after_;  // [2]
    DeviceBuffer<int>
        d_samples_; // where the indices to sample a molecule come from, currently fixed to a single sample
    DeviceBuffer<RealType> d_quaternions_;  // Normal noise for uniform random rotations
    DeviceBuffer<RealType> d_translations_; // Uniform noise for translation + the check
    DeviceBuffer<size_t> d_num_accepted_;
    DeviceBuffer<int> d_target_mol_atoms_;
    DeviceBuffer<int> d_target_mol_offsets_;

    curandGenerator_t cr_rng_;

    void compute_initial_weights(const int N, double *d_coords, double *d_box, cudaStream_t stream);

    void
    compute_incremental_weights(const int N, const bool scale, double *d_coords, double *d_box, cudaStream_t stream);

public:
    BDExchangeMove(
        const int N,
        const std::vector<std::vector<int>> &target_mols,
        const std::vector<double> &params,
        const double temperature,
        const double nb_beta,
        const double cutoff,
        const int seed,
        const int proposals_per_move);

    ~BDExchangeMove();

    virtual void move_device(
        const int N,
        double *d_coords, // [N, 3]
        double *d_box,    // [3, 3]
        cudaStream_t stream);

    virtual std::array<std::vector<double>, 2> move_host(const int N, const double *h_coords, const double *h_box);

    virtual double log_probability_host();

    size_t n_proposed() const { return num_attempted_; }

    size_t n_accepted() const;

    double acceptance_fraction() const {
        return static_cast<double>(this->n_accepted()) / static_cast<double>(this->n_proposed());
    }
};

} // namespace timemachine
