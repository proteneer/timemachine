#include "summed_potential.hpp"
#include <memory>
#include <numeric>
#include <stdexcept>

namespace timemachine {

SummedPotential::SummedPotential(
    const std::vector<std::shared_ptr<Potential>> potentials, const std::vector<int> params_sizes, const bool serial)
    : potentials_(potentials), params_sizes_(params_sizes),
      P_(std::accumulate(params_sizes.begin(), params_sizes.end(), 0)), serial_(serial) {
    if (potentials_.size() != params_sizes_.size()) {
        throw std::runtime_error("number of potentials != number of parameter sizes");
    }
};

const std::vector<std::shared_ptr<Potential>> &SummedPotential::get_potentials() { return potentials_; }

const std::vector<int> &SummedPotential::get_parameter_sizes() { return params_sizes_; }

void SummedPotential::execute_device(
    const int N,
    const int P,
    const double *d_x,
    const double *d_p,
    const double *d_box,
    unsigned long long *d_du_dx,
    unsigned long long *d_du_dp,
    unsigned long long *d_u,
    cudaStream_t stream) {

    if (P != P_) {
        throw std::runtime_error(
            "SummedPotential::execute_device(): expected " + std::to_string(P_) + " parameters, got " +
            std::to_string(P));
    }

    int offset = 0;
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
        potentials_[i]->execute_device(
            N,
            params_sizes_[i],
            d_x,
            d_p + offset,
            d_box,
            d_du_dx,
            d_du_dp == nullptr ? nullptr : d_du_dp + offset,
            d_u,
            pot_stream);

        offset += params_sizes_[i];
        if (!serial_) {
            manager_.sync_to(i, stream);
        }
    }
};

void SummedPotential::du_dp_fixed_to_float(
    const int N, const int P, const unsigned long long *du_dp, double *du_dp_float) {

    int offset = 0;

    for (auto i = 0; i < potentials_.size(); i++) {
        potentials_[i]->du_dp_fixed_to_float(N, params_sizes_[i], du_dp + offset, du_dp_float + offset);
        offset += params_sizes_[i];
    }
}

} // namespace timemachine
