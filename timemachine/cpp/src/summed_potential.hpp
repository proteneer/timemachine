#pragma once

#include "bound_potential.hpp"
#include "potential.hpp"

namespace timemachine {

class SummedPotential : public Potential {

private:
    const std::vector<BoundPotential *> bps_;

public:
    SummedPotential(const std::vector<BoundPotential *> bps);

    virtual void execute_device(
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
        cudaStream_t stream) override;
};

} // namespace timemachine
