#pragma once

#include "bd_exchange_move.hpp"
#include "curand_kernel.h"
#include "device_buffer.hpp"
#include "pinned_host_buffer.hpp"
#include <array>
#include <vector>

namespace timemachine {

// TIBDExchangeMove uses targeted insertion and biased deletion to move into a sphere around a set of ligand atoms. The reference implementation
// is in timemachine/md/exchange/exchange_mover.py::TIBDExchangeMove
// Refer to timemachine/cpp/src/bd_exchange_move.hpp::BDExchangeMove for the definitions of the terminology
template <typename RealType> class TIBDExchangeMove : public BDExchangeMove<RealType> {

protected:
    const RealType radius_;
    const RealType inner_volume_;

    DeviceBuffer<curandState_t> d_rand_states_;

    DeviceBuffer<int> d_inner_mols_count_;    // [1]
    DeviceBuffer<int> d_identify_indices_;    // [this->num_target_mols_]
    DeviceBuffer<int> d_partitioned_indices_; // [this->num_target_mols_]
    DeviceBuffer<char> d_temp_storage_buffer_;
    size_t temp_storage_bytes_;

    DeviceBuffer<RealType> d_center_; // [3]
    // Uniform noise with the first element used for deciding directionality of insertion
    // and the second element is used for comparison against the acceptance rate in the Metropolis-Hastings check
    DeviceBuffer<RealType> d_uniform_noise_buffer_; // [2 * this->batch_size_ * this->steps_per_move_]
    DeviceBuffer<int> d_targeting_inner_vol_;       // [1]

    DeviceBuffer<int> d_ligand_idxs_;
    DeviceBuffer<RealType> d_src_log_weights_;  // [this->num_target_mols_ * this->batch_size_]
    DeviceBuffer<RealType> d_dest_log_weights_; // [this->num_target_mols_ * this->batch_size_]
    DeviceBuffer<int> d_inner_flags_;           // [this->num_target_mols_]
    DeviceBuffer<RealType> d_box_volume_;       // [1]

private:
    DeviceBuffer<RealType> d_selected_translations_;   // [this->batch_size_, 3] The translation selected to run
    DeviceBuffer<int> d_sample_after_segment_offsets_; // [this->batch_size_ + 1]
    DeviceBuffer<int> d_weights_before_counts_;        // [this->batch_size_]
    DeviceBuffer<int> d_weights_after_counts_;         // [this->batch_size_]

    DeviceBuffer<RealType> d_lse_max_src_;     // [this->batch_size, this->num_target_mols_]
    DeviceBuffer<RealType> d_lse_exp_sum_src_; // [this->batch_size, this->num_target_mols_]

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
        const int proposals_per_move,
        const int interval,
        const int batch_size);

    ~TIBDExchangeMove();

    void move(
        const int N,
        double *d_coords, // [N, 3]
        double *d_box,    // [3, 3]
        cudaStream_t stream) override;

    std::array<std::vector<double>, 2> move_host(const int N, const double *h_coords, const double *h_box) override;

    double log_probability_host() override;
    double raw_log_probability_host() override;
};

} // namespace timemachine
