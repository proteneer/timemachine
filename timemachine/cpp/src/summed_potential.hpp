#pragma once

#include "potential.hpp"

namespace timemachine {

class SummedPotential : public Potential {

private:
    Potential &u_a;
    Potential &u_b;
    int P_a; // number of parameters for first potential

public:
    SummedPotential(Potential &u_a, Potential &u_b, const int P_a);

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
