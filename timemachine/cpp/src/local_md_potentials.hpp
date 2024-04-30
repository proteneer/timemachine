#pragma once
#include "device_buffer.hpp"
#include "pinned_host_buffer.hpp"
#include <memory>
#include <vector>

#include "bound_potential.hpp"
#include "curand.h"
#include "flat_bottom_bond.hpp"
#include "local_md_utils.hpp"
#include "log_flat_bottom_bond.hpp"

namespace timemachine {

class LocalMDPotentials {

public:
    LocalMDPotentials(
        const int N,
        const std::vector<std::shared_ptr<BoundPotential>> &bps,
        bool freeze_reference = true,
        double temperature = 0.0);

    ~LocalMDPotentials();

    unsigned int *get_free_idxs();

    void setup_from_idxs(
        double *d_x_t_,
        double *d_box_t,
        const std::vector<int> &local_idxs,
        const int seed,
        const double radius,
        const double k,
        cudaStream_t stream);

    void setup_from_selection(
        const int reference_idx,
        const std::vector<int> &selection_idxs,
        const double radius,
        const double k,
        cudaStream_t stream);

    std::vector<std::shared_ptr<BoundPotential>> get_potentials();

    void reset_potentials(cudaStream_t stream);

    const bool freeze_reference;
    const double temperature;

private:
    const int N_;
    std::size_t temp_storage_bytes_;
    int num_allpairs_idxs_;

    std::vector<std::shared_ptr<BoundPotential>> all_potentials_;
    std::shared_ptr<BoundPotential> ixn_group_;
    std::shared_ptr<BoundPotential> nonbonded_bp_;
    // Restraint for the free particles to the reference particle
    std::shared_ptr<FlatBottomBond<float>> free_restraint_;
    std::shared_ptr<BoundPotential> bound_free_restraint_;

    // Restraint for the frozen particles to the reference particle
    std::shared_ptr<LogFlatBottomBond<float>> frozen_restraint_;
    std::shared_ptr<BoundPotential> bound_frozen_restraint_;

    DeviceBuffer<int> d_restraint_pairs_;
    DeviceBuffer<double> d_bond_params_;

    DeviceBuffer<float> d_probability_buffer_;

    DeviceBuffer<unsigned int> d_free_idxs_;
    DeviceBuffer<unsigned int> d_temp_idxs_; // Where intersection of free/frozen and initial atom indices are stored
    DeviceBuffer<unsigned int> d_all_pairs_idxs_; // Where initial atom indices, of NonbondedAllPairs, are stored
    DeviceBuffer<char> d_temp_storage_buffer_;

    DeviceBuffer<unsigned int> d_row_idxs_;
    DeviceBuffer<unsigned int> d_col_idxs_;

    // Pinned memory for getting lengths of index arrays
    PinnedHostBuffer<int> p_num_selected_;

    DeviceBuffer<int> d_num_selected_buffer_;

    curandGenerator_t cr_rng_;

    void _setup_free_idxs_given_reference_idx(
        const unsigned int reference_idx, const double radius, const double k, cudaStream_t stream);
};

} // namespace timemachine
