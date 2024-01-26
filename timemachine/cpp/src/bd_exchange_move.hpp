#pragma once

#include "device_buffer.hpp"
#include "mover.hpp"
#include "nonbonded_mol_energy.hpp"
#include "segmented_sumexp.hpp"
#include "segmented_weighted_random_sampler.hpp"
#include <array>
#include <vector>

namespace timemachine {

/* BDExchangeMove uses biased deletion to move waters randomly in a box. The reference implementation
* is in timemachine/md/exchange/exchange_mover.py::BDExchangeMove
*
* Terminology:
* - Step: An internal iteration of a call to BDExchangeMove::move
* - Proposal: A rotation/translation of a molecule to be evaluated for acceptance
* - Move: A move is made up a set of steps that make/accept batches of proposals
*/
template <typename RealType> class BDExchangeMove : public Mover {

protected:
    // Amount of random values to generate at a time
    static const int QUATERNIONS_PER_STEP = 4;
    const int N_;
    // Number of atom in all mols
    // All molecules are currently expected to have same number of atoms (typically 3 for waters)
    // Done to avoid having to determine the size of the sample and allows us to test ion sampling by having
    // two different BDExchangemoves
    const int mol_size_;
    const int steps_per_move_;
    const int num_target_mols_;
    const RealType nb_beta_;
    const RealType beta_; // 1 / kT
    const RealType cutoff_squared_;
    const int proposals_per_step_;
    size_t num_attempted_;
    NonbondedMolEnergyPotential<RealType> mol_potential_;
    SegmentedWeightedRandomSampler<RealType> sampler_;
    SegmentedSumExp<RealType> logsumexp_;
    // Buffer for evaluating moves without touching the original coords
    DeviceBuffer<double> d_intermediate_coords_;             // [proposals_per_step_, mol_size_, 3]
    DeviceBuffer<double> d_params_;                          // [N, PARAMS_PER_ATOM]
    DeviceBuffer<__int128> d_mol_energy_buffer_;             // [proposals_per_step_, num_target_mols_]
    DeviceBuffer<RealType> d_sample_per_atom_energy_buffer_; // [proposals_per_step_, mol_size_ * N]
    DeviceBuffer<int> d_atom_idxs_;                          // [num_target_mols_, mol_size_]
    DeviceBuffer<int> d_mol_offsets_;                        // [num_target_mols_ + 1]
    DeviceBuffer<RealType> d_log_weights_before_;            // [num_target_mols_]
    DeviceBuffer<RealType> d_log_weights_after_;             // [proposals_per_step_, num_target_mols_]

    // Arrays used for computing logsumexp, split into max component and the sum component
    DeviceBuffer<RealType> d_lse_max_before_;     // [1]
    DeviceBuffer<RealType> d_lse_exp_sum_before_; // [1]
    DeviceBuffer<RealType> d_lse_max_after_;      // [proposals_per_step_]
    DeviceBuffer<RealType> d_lse_exp_sum_after_;  // [proposals_per_step_]

    DeviceBuffer<int> d_samples_;            // [proposals_per_step_] The indices of the molecules to make proposals for
    DeviceBuffer<RealType> d_quaternions_;   // Normal noise for uniform random rotations
    DeviceBuffer<size_t> d_num_accepted_;    // [1]
    DeviceBuffer<int> d_target_mol_atoms_;   // [proposals_per_step_, mol_size_]
    DeviceBuffer<int> d_target_mol_offsets_; // [num_target_mols + 1]
    DeviceBuffer<__int128> d_intermediate_sample_weights_;
    DeviceBuffer<RealType> d_sample_noise_; // Noise to use for selecting molecules
    DeviceBuffer<RealType>
        d_sampling_intermediate_; // [proposals_per_step_, num_target_mols_] Intermediate buffer for weighted sampling
    DeviceBuffer<RealType> d_translations_;       // Uniform noise for translation + the check
    DeviceBuffer<int> d_sample_segments_offsets_; // Segment offsets for the sampler

    curandGenerator_t cr_rng_quat_;
    curandGenerator_t cr_rng_translations_;
    curandGenerator_t cr_rng_samples_;

    void compute_initial_weights(const int N, double *d_coords, double *d_box, cudaStream_t stream);

    void compute_incremental_weights(
        const int N,
        const bool scale,
        const double *d_box,
        const double *d_coords,
        const RealType *d_quaternions,
        const RealType *d_translations,
        cudaStream_t stream);

    BDExchangeMove(
        const int N,
        const std::vector<std::vector<int>> &target_mols,
        const std::vector<double> &params,
        const double temperature,
        const double nb_beta,
        const double cutoff,
        const int seed,
        const int proposals_per_move,
        const int interval,
        const int proposals_per_step,
        const int translation_buffer_size);

public:
    BDExchangeMove(
        const int N,
        const std::vector<std::vector<int>> &target_mols,
        const std::vector<double> &params,
        const double temperature,
        const double nb_beta,
        const double cutoff,
        const int seed,
        const int proposals_per_move,
        const int interval,
        const int proposals_per_step);

    ~BDExchangeMove();

    virtual void move(
        const int N,
        double *d_coords, // [N, 3]
        double *d_box,    // [3, 3]
        cudaStream_t stream) override;

    virtual double log_probability_host();
    virtual double raw_log_probability_host();

    size_t n_proposed() const { return num_attempted_; }

    size_t n_accepted() const;

    std::vector<double> get_params();

    void set_params(const std::vector<double> &params);

    void set_params_device(const int size, const double *d_p, const cudaStream_t stream);

    double acceptance_fraction() const {
        return static_cast<double>(this->n_accepted()) / static_cast<double>(this->n_proposed());
    }
};

} // namespace timemachine
