#pragma once
#include "device_buffer.hpp"
#include "pinned_host_buffer.hpp"
#include <memory>
#include <vector>

#include "bound_potential.hpp"
#include "curand.h"
#include "flat_bottom_bond.hpp"
#include "local_md_utils.hpp"

namespace timemachine {

class LocalMDConfig {

public:
    LocalMDConfig(const int N, const std::vector<std::shared_ptr<BoundPotential>> bps);

    ~LocalMDConfig();

    DeviceBuffer<unsigned int> *get_free_idxs();

    void setup_from_idxs(
        double *d_x_t_,
        double *d_box_t,
        const std::vector<int> &local_idxs,
        const double temperature,
        const int seed,
        const double radius,
        const double k,
        const cudaStream_t stream);

    std::vector<std::shared_ptr<BoundPotential>> get_potentials();

    void reset(const cudaStream_t stream);

private:
    const int N_;
    std::size_t temp_storage_bytes_;

    std::vector<std::shared_ptr<BoundPotential>> all_potentials_;
    std::shared_ptr<BoundPotential> ixn_group_;
    std::shared_ptr<BoundPotential> nonbonded_bp_;
    std::shared_ptr<FlatBottomBond<double>> restraint_;
    std::shared_ptr<BoundPotential> bound_restraint_;

    DeviceBuffer<int> restraints_;
    DeviceBuffer<double> bond_params_;

    DeviceBuffer<float> probability_buffer_;
    DeviceBuffer<unsigned int> d_free_idxs_;
    std::unique_ptr<DeviceBuffer<char>> d_temp_storage_buffer_;

    DeviceBuffer<unsigned int> d_row_idxs_;
    DeviceBuffer<unsigned int> d_col_idxs_;

    // Pinned memory for getting lengths of indice arrays
    PinnedHostBuffer<int> p_num_selected_;
    DeviceBuffer<int> num_selected_buffer_;

    curandGenerator_t cr_rng_;

    void _setup_free(const unsigned int reference_idx, const double radius, const double k, const cudaStream_t stream);
};

} // namespace timemachine
