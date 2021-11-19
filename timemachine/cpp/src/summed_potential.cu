#include "summed_potential.hpp"
#include <stdexcept>

namespace timemachine {

SummedPotential::SummedPotential(std::vector<Potential *> potentials, std::vector<int> param_sizes)
    : potentials_(potentials), param_sizes_(param_sizes) {
    if (potentials_.size() != param_sizes_.size()) {
        throw std::runtime_error("number of potentials != number of parameter sets");
    }
};

void SummedPotential::execute_device(
    const int N,
    const int P,
    const double *d_x,
    const double *d_p,
    const double *d_box,
    const double lambda,
    unsigned long long *d_du_dx,
    double *d_du_dp,
    unsigned long long *d_du_dl,
    unsigned long long *d_u,
    cudaStream_t stream) {

    int offset = 0;

    for (int i = 0; i < potentials_.size(); i++) {

        potentials_[i]->execute_device(
            N,
            param_sizes_[i],
            d_x,
            d_p + offset,
            d_box,
            lambda,
            d_du_dx,
            d_du_dp == nullptr ? d_du_dp : d_du_dp + offset,
            d_du_dl,
            d_u,
            stream);

        offset += param_sizes_[i];
    }
};

} // namespace timemachine
