#pragma once

#include "bd_exchange_move.hpp"
#include "curand_kernel.h"
#include "device_buffer.hpp"
#include "logsumexp.hpp"
#include "nonbonded_mol_energy.hpp"
#include "pinned_host_buffer.hpp"
#include "weighted_random_sampler.hpp"
#include <array>
#include <vector>

namespace timemachine {

// TIBDExchangeMove uses targeted insertion and biased deletion to move into a sphere around a set of ligand atoms. The reference implementation
// is in timemachine/md/exchange/exchange_mover.py::TIBDExchangeMove
template <typename RealType> class TIBDExchangeMove : public BDExchangeMove<RealType> {

protected:
    const RealType radius_;
    const RealType inner_volume_;

    DeviceBuffer<curandState_t> d_rand_states_;

    DeviceBuffer<int> d_inner_mols_count_; // [1]
    DeviceBuffer<int> d_inner_mols_;       // [num_target_mols_]
    DeviceBuffer<int> d_outer_mols_count_; // [1]
    DeviceBuffer<int> d_outer_mols_;       // [num_target_mols_]
    // Array to sort the inner/outer indices to ensure that the results are deterministic
    DeviceBuffer<int> d_sorted_indices_; // [num_target_mols_]
    DeviceBuffer<char> d_sort_storage_;
    size_t sort_storage_bytes_;

    DeviceBuffer<RealType> d_center_;      // [3]
    DeviceBuffer<RealType> d_translation_; // [3]
    // Uniform noise for determining where to insert and whether to accept the move
    DeviceBuffer<RealType> d_acceptance_;     // [2]
    DeviceBuffer<int> d_targeting_inner_vol_; // [1]

    DeviceBuffer<int> d_ligand_idxs_;
    DeviceBuffer<RealType> d_src_weights_;        // [num_target_mols_]
    DeviceBuffer<RealType> d_dest_weights_;       // [num_target_mols_]
    DeviceBuffer<RealType> d_box_volume_;         // [1]
    PinnedHostBuffer<int> p_inner_count_;         // [1]
    PinnedHostBuffer<int> p_targeting_inner_vol_; // [1]

    cudaEvent_t host_copy_event_;

public:
    TIBDExchangeMove(
        const int N,
        const std::vector<int> ligand_idxs,
        const std::vector<std::vector<int>> &target_mols,
        const std::vector<double> &params,
        const double temperature,
        const double nb_beta,
        const double cutoff,
        const double radius,
        const int seed,
        const int proposals_per_move);

    ~TIBDExchangeMove();

    void move_device(
        const int N,
        double *d_coords, // [N, 3]
        double *d_box,    // [3, 3]
        cudaStream_t stream) override;

    std::array<std::vector<double>, 2> move_host(const int N, const double *h_coords, const double *h_box) override;

    double log_probability_host() override;
};

} // namespace timemachine
