#pragma once

#include <vector>

#include "bound_potential.hpp"
#include "integrator.hpp"
#include "streamed_potential_runner.hpp"

namespace timemachine {

class LangevinIntegrator : public Integrator {

private:
    const int N_;
    const double temperature_;
    const double dt_;
    const double friction_;
    double ca_;

    // The offset into the current batch of noise
    int noise_offset_;

    double *d_cbs_;
    double *d_ccs_;
    double *d_noise_;
    unsigned long long *d_du_dx_;

    curandGenerator_t cr_rng_;

    StreamedPotentialRunner runner_;

public:
    LangevinIntegrator(int N, const double *masses, double temperature, double dt, double friction, int seed);

    virtual ~LangevinIntegrator();

    double get_temperature();

    virtual void step_fwd(
        std::vector<std::shared_ptr<BoundPotential>> &bps,
        double *d_x_t,
        double *d_v_t,
        double *d_box_t_,
        unsigned int *d_idxs,
        cudaStream_t stream) override;

    virtual void initialize(
        std::vector<std::shared_ptr<BoundPotential>> &bps,
        double *d_x_t,
        double *d_v_t,
        double *d_box_t,
        unsigned int *d_idxs,
        cudaStream_t stream) override;

    virtual void finalize(
        std::vector<std::shared_ptr<BoundPotential>> &bps,
        double *d_x_t,
        double *d_v_t,
        double *d_box_t,
        unsigned int *d_idxs,
        cudaStream_t stream) override;
};

} // end namespace timemachine
