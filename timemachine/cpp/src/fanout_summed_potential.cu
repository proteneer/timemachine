#include "fanout_summed_potential.hpp"
#include <memory>

namespace timemachine {

FanoutSummedPotential::FanoutSummedPotential(
    const std::vector<std::shared_ptr<Potential>> potentials, const bool serial)
    : potentials_(potentials), serial_(serial){};

const std::vector<std::shared_ptr<Potential>> &FanoutSummedPotential::get_potentials() { return potentials_; }

void FanoutSummedPotential::execute_device(
    const int N,
    const int P,
    const double *d_x,
    const double *d_p,
    const double *d_box,
    unsigned long long *d_du_dx,
    unsigned long long *d_du_dp,
    unsigned long long *d_u,
    cudaStream_t stream) {

    if (!serial_) {
        for (auto i = 0; i < potentials_.size(); i++) {
            // Always sync the new streams with the incoming stream to ensure that the state
            // of the incoming buffers are valid
            manager_.sync_from(i, stream);
        }
    }
    cudaStream_t pot_stream = stream;
    for (auto i = 0; i < potentials_.size(); i++) {
        if (!serial_) {
            pot_stream = manager_.get_stream(i);
        }
        potentials_[i]->execute_device(N, P, d_x, d_p, d_box, d_du_dx, d_du_dp, d_u, pot_stream);
        if (!serial_) {
            manager_.sync_to(i, stream);
        }
    }
};

void FanoutSummedPotential::du_dp_fixed_to_float(
    const int N, const int P, const unsigned long long *du_dp, double *du_dp_float) {

    if (!potentials_.empty()) {
        potentials_[0]->du_dp_fixed_to_float(N, P, du_dp, du_dp_float);
    }
}

} // namespace timemachine
