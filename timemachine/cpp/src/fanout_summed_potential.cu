#include "energy_accumulation.hpp"
#include "fanout_summed_potential.hpp"
#include "gpu_utils.cuh"
#include "nonbonded_common.hpp"
#include <memory>

namespace timemachine {

FanoutSummedPotential::FanoutSummedPotential(
    const std::vector<std::shared_ptr<Potential>> potentials, const bool parallel)
    : potentials_(potentials), parallel_(parallel), d_u_buffer_(potentials_.size()) {};

const std::vector<std::shared_ptr<Potential>> &FanoutSummedPotential::get_potentials() { return potentials_; }

void FanoutSummedPotential::execute_device(
    const int N,
    const int P,
    const double *d_x,
    const double *d_p,
    const double *d_box,
    unsigned long long *d_du_dx,
    unsigned long long *d_du_dp,
    __int128 *d_u,
    cudaStream_t stream) {

    if (d_u) {
        gpuErrchk(cudaMemsetAsync(d_u_buffer_.data, 0, d_u_buffer_.size(), stream));
    }

    if (parallel_) {
        for (auto i = 0; i < potentials_.size(); i++) {
            // Always sync the new streams with the incoming stream to ensure that the state
            // of the incoming buffers are valid
            manager_.sync_from(i, stream);
        }
    }
    cudaStream_t pot_stream = stream;
    for (auto i = 0; i < potentials_.size(); i++) {
        if (parallel_) {
            pot_stream = manager_.get_stream(i);
        }
        potentials_[i]->execute_device(
            N, P, d_x, d_p, d_box, d_du_dx, d_du_dp, d_u == nullptr ? nullptr : d_u_buffer_.data + i, pot_stream);
        if (parallel_) {
            manager_.sync_to(i, stream);
        }
    }
    if (d_u) {
        accumulate_energy(potentials_.size(), d_u_buffer_.data, d_u, stream);
    }
};

void FanoutSummedPotential::du_dp_fixed_to_float(
    const int N, const int P, const unsigned long long *du_dp, double *du_dp_float) {

    if (!potentials_.empty()) {
        potentials_[0]->du_dp_fixed_to_float(N, P, du_dp, du_dp_float);
    }
}

} // namespace timemachine
