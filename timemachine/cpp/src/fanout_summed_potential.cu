#include "fanout_summed_potential.hpp"
#include <memory>

namespace timemachine {

FanoutSummedPotential::FanoutSummedPotential(const std::vector<std::shared_ptr<Potential>> potentials)
    : potentials_(potentials){};

const std::vector<std::shared_ptr<Potential>> &FanoutSummedPotential::get_potentials() { return potentials_; }

void FanoutSummedPotential::execute_device(
    const int N,
    const int P,
    const double *d_x,
    const double *d_p,
    const double *d_box,
    const double lambda,
    unsigned long long *d_du_dx,
    unsigned long long *d_du_dp,
    unsigned long long *d_du_dl,
    unsigned long long *d_u,
    cudaStream_t stream) {

    for (auto potential : potentials_) {
        potential->execute_device(N, P, d_x, d_p, d_box, lambda, d_du_dx, d_du_dp, d_du_dl, d_u, stream);
    }
};

void FanoutSummedPotential::du_dp_fixed_to_float(
    const int N, const int P, const unsigned long long *du_dp, double *du_dp_float) {

    if (!potentials_.empty()) {
        potentials_[0]->du_dp_fixed_to_float(N, P, du_dp, du_dp_float);
    }
}

} // namespace timemachine
