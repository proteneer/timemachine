#pragma once

#include "curand.h"

namespace timemachine {

class Integrator {

public:

    virtual ~Integrator() {};

    virtual void step_fwd(
        double *d_x_t,
        double *d_v_t,
        unsigned long long *d_du_dx_t,
        double *d_box_t_
    ) = 0;

};

class LangevinIntegrator : public Integrator {

private:

    double dt_;
    double N_;
    double ca_;
    double *d_cbs_;
    double *d_ccs_;
    double *d_noise_;

    curandGenerator_t  cr_rng_;

public:

    LangevinIntegrator(
        int N,
        double dt,
        double ca,
        const double *h_cbs,
        const double *h_ccs,
        int seed
    );

    virtual ~LangevinIntegrator();

    virtual void step_fwd(
        double *d_x_t,
        double *d_v_t,
        unsigned long long *d_du_dx_t,
        double *d_box_t_
    ) override;

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