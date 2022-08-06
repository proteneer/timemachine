#pragma once

#include "bound_potential.hpp"
#include <vector>

namespace timemachine {

class Integrator {

public:
    virtual ~Integrator(){};

    virtual void step_fwd(
        std::vector<BoundPotential *> bps,
        double lamb,
        double *d_x_t,
        double *d_v_t,
        double *d_box_t,
        unsigned long long *d_du_dl,
        cudaStream_t stream) = 0;

    virtual void finalize(
        std::vector<BoundPotential *> bps,
        double lamb,
        double *d_x_t,
        double *d_v_t,
        double *d_box_t,
        cudaStream_t stream) = 0;
};

// template<typename RealType>
// void step_forward(
//     int N,
//     int D,
//     const RealType ca,
//     const RealType *d_coeff_bs,
//     const RealType *d_coeff_cs,
//     const RealType *d_noise_buf,
//     const RealType *d_x_old,
//     const RealType *d_v_old,
//     const unsigned long long *d_dE_dx,
//     const RealType dt,
//     RealType *d_x_new,
//     RealType *d_v_new);

} // end namespace timemachine
