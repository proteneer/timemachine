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

private:
    const int N_;
    const int num_target_mols_;
    const RealType beta_; // 1 / kT
    NonbondedMolEnergyPotential<RealType> mol_potential_;
    WeightedRandomSampler<RealType> sampler_;
    LogSumExp<RealType> logsumexp_;
    // Buffer for evaluating moves without touching the original coords
    const DeviceBuffer<double> d_intermediate_coords_;
    const DeviceBuffer<double> d_params_;
    const DeviceBuffer<__int128> d_mol_energy_buffer_;
    const DeviceBuffer<int> d_mol_offsets_;
    const DeviceBuffer<RealType> d_log_weights_;
    const DeviceBuffer<RealType> d_log_probabilities_before_;
    const DeviceBuffer<RealType> d_log_probabilities_after_;
    const DeviceBuffer<RealType> d_log_sum_exp_before_; // [2]
    const DeviceBuffer<RealType> d_log_sum_exp_after_;  // [2]
    const DeviceBuffer<int>
        d_samples_; // where the indices to sample a molecule come from, currently fixed to a single sample
    const DeviceBuffer<RealType> d_quaternions_;  // Normal noise for uniform random rotations
    const DeviceBuffer<RealType> d_translations_; // Uniform noise for translation + the check

    curandGenerator_t cr_rng_;

public:
    BDExchangeMove(
        const int N,
        const std::vector<std::vector<int>> &target_mols,
        const std::vector<double> &params,
        const double temperature,
        const double nb_beta,
        const double cutoff,
        const int seed);

    ~BDExchangeMove();

    void move_device(
        const int N,
        const int num_moves,
        double *d_coords, // [N, 3]
        double *d_box,    // [3, 3]
        cudaStream_t stream);

    std::array<std::vector<double>, 2>
    move_host(const int N, const int num_moves, const double *h_coords, const double *h_box);

    double log_probability_host();
};

} // namespace timemachine
