#pragma once

#include "potential.hpp"
#include <vector>

namespace timemachine {

class SummedPotential : public Potential {

private:
    const std::vector<Potential *> potentials_;
    const std::vector<int> param_sizes_;

public:
    SummedPotential(std::vector<Potential *> potentials, std::vector<int> param_sizes);

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
