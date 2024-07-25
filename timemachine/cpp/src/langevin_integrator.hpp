#pragma once

#include <vector>

#include "bound_potential.hpp"
#include "integrator.hpp"
#include "streamed_potential_runner.hpp"

namespace timemachine {

template <typename RealType> class LangevinIntegrator : public Integrator {

private:
    const int N_;
    const double temperature_;
    const RealType dt_;
    const double friction_;
    RealType ca_;

    // The offset into the current batch of noise
    int noise_offset_;

    RealType *d_cbs_;
    RealType *d_ccs_;
    RealType *d_noise_;
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
