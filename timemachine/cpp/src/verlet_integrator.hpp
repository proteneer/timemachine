#pragma once

#include <vector>

#include "bound_potential.hpp"
#include "integrator.hpp"

namespace timemachine {

class VelocityVerletIntegrator : public Integrator {

private:
    const int N_;
    const double dt_;
    bool initialized_;
    double *d_cbs_;
    unsigned long long *d_du_dx_;

public:
    VelocityVerletIntegrator(int N, double dt, const double *h_cbs);

    virtual ~VelocityVerletIntegrator();

    virtual void step_fwd(
        std::vector<BoundPotential *> bps,
        double lamb,
        double *d_x_t,
        double *d_v_t,
        double *d_box_t_,
        unsigned long long *d_du_dl,
        cudaStream_t stream) override;

    virtual void finalize(
        std::vector<BoundPotential *> bps,
        double lamb,
        double *d_x_t,
        double *d_v_t,
        double *d_box_t,
        cudaStream_t stream) override;
};

} // end namespace timemachine
