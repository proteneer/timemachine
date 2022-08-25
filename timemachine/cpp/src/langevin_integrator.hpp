#pragma once

#include <vector>

#include "bound_potential.hpp"
#include "integrator.hpp"

namespace timemachine {

class LangevinIntegrator : public Integrator {

private:
    const int N_;
    const double temperature_;
    const double dt_;
    const double friction_;
    double ca_;
    double *d_cbs_;
    double *d_ccs_;
    double *d_noise_;
    unsigned long long *d_du_dx_;

    curandGenerator_t cr_rng_;

public:
    LangevinIntegrator(int N, const double *masses, double temperature, double dt, double friction, int seed);

    virtual ~LangevinIntegrator();

    virtual void step_fwd(
        std::vector<BoundPotential *> &bps,
        double lamb,
        double *d_x_t,
        double *d_v_t,
        double *d_box_t,
        unsigned long long *d_du_dl,
        unsigned int *d_idxs,
        cudaStream_t stream) override;

    virtual void initialize(
        std::vector<BoundPotential *> &bps,
        double lamb,
        double *d_x_t,
        double *d_v_t,
        double *d_box_t,
        unsigned int *d_idxs,
        cudaStream_t stream) override;

    virtual void finalize(
        std::vector<BoundPotential *> &bps,
        double lamb,
        double *d_x_t,
        double *d_v_t,
        double *d_box_t,
        unsigned int *d_idxs,
        cudaStream_t stream) override;
};

} // end namespace timemachine
