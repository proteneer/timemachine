#pragma once

#include "potential.hpp"

#include <memory>
#include <typeinfo>

namespace timemachine {

class LambdaPotential : public Potential {

private:

    std::shared_ptr<Potential> u_;

    unsigned long long *d_du_dx_buffer_;
    double *d_du_dp_buffer_;
    double *d_du_dl_buffer_;
    double *d_u_buffer_;

    int sign_;

public: 

    LambdaPotential(
        std::shared_ptr<Potential> u,
        int N,
        int P,
        int sign
    );

    ~LambdaPotential();

    virtual void execute_device(
        const int N,
        const int P,
        const double *d_x,
        const double *d_p,
        const double *d_box,
        const double lambda,
        unsigned long long *d_du_dx,
        double *d_du_dp,
        double *d_du_dl,
        double *d_u,
        cudaStream_t stream
    ) override;

};

}
