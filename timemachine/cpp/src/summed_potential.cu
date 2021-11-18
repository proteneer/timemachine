#include "summed_potential.hpp"

namespace timemachine {

SummedPotential::SummedPotential(const std::vector<BoundPotential *> bps) : bps_(bps){};

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

    for (int i = 0; i < bps_.size(); i++) {
        bps_[i]->execute_device(N, d_x, d_box, lambda, d_du_dx, d_du_dp, d_du_dl, d_u, stream);
    }
};

} // namespace timemachine
