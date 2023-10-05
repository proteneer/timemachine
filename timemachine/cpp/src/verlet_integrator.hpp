#pragma once

#include <vector>

#include "bound_potential.hpp"
#include "integrator.hpp"
#include "streamed_potential_runner.hpp"

namespace timemachine {

class VelocityVerletIntegrator : public Integrator {

private:
    const int N_;
    const double dt_;
    bool initialized_;
    double *d_cbs_;
    unsigned long long *d_du_dx_;
    StreamedPotentialRunner runner_;

public:
    VelocityVerletIntegrator(int N, double dt, const double *h_cbs);

    virtual ~VelocityVerletIntegrator();

    virtual void step_fwd(
        std::vector<std::shared_ptr<BoundPotential>> &bps,
        CoordsType *d_x_t,
        double *d_v_t,
        CoordsType *d_box_t,
        unsigned int *d_idxs,
        cudaStream_t stream) override;

    virtual void initialize(
        std::vector<std::shared_ptr<BoundPotential>> &bps,
        CoordsType *d_x_t,
        double *d_v_t,
        CoordsType *d_box_t,
        unsigned int *d_idxs,
        cudaStream_t stream) override;

    virtual void finalize(
        std::vector<std::shared_ptr<BoundPotential>> &bps,
        CoordsType *d_x_t,
        double *d_v_t,
        CoordsType *d_box_t,
        unsigned int *d_idxs,
        cudaStream_t stream) override;
};

} // end namespace timemachine
