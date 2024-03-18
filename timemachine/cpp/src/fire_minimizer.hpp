#pragma once

#include <vector>

#include "bound_potential.hpp"
#include "integrator.hpp"
#include "streamed_potential_runner.hpp"

namespace timemachine {

template <typename RealType> class FireMinimizer : public Integrator {

private:
    const int N_;
    const RealType dt_;
    unsigned long long *d_du_dx_;
    StreamedPotentialRunner runner_;

public:
    FireMinimizer(int N, double dt);

    virtual ~FireMinimizer();

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
