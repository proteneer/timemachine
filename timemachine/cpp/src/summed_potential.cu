#include "summed_potential.hpp"
#include <memory>
#include <stdexcept>

namespace timemachine {

SummedPotential::SummedPotential(std::vector<std::shared_ptr<Potential>> potentials, std::vector<int> params_sizes)
    : potentials_(potentials), params_sizes_(params_sizes) {
    if (potentials_.size() != params_sizes_.size()) {
        throw std::runtime_error("number of potentials != number of parameter sizes");
    }
};

const std::vector<std::shared_ptr<Potential>> &SummedPotential::get_potentials() { return potentials_; }

void SummedPotential::execute_device(
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

    int offset = 0;

    for (int i = 0; i < potentials_.size(); i++) {

        potentials_[i]->execute_device(
            N,
            params_sizes_[i],
            d_x,
            d_p + offset,
            d_box,
            lambda,
            d_du_dx,
            d_du_dp == nullptr ? nullptr : d_du_dp + offset,
            d_du_dl,
            d_u,
            stream);

        offset += params_sizes_[i];
    }
};

void SummedPotential::du_dp_fixed_to_float(
    const int N, const int P, const unsigned long long *du_dp, double *du_dp_float) {

    int offset = 0;

    for (int i = 0; i < potentials_.size(); i++) {
        potentials_[i]->du_dp_fixed_to_float(N, params_sizes_[i], du_dp + offset, du_dp_float + offset);
        offset += params_sizes_[i];
    }
}

} // namespace timemachine
