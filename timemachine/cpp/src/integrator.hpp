#pragma once

#include "bound_potential.hpp"
#include <vector>

namespace timemachine {

class Integrator {

public:
    virtual ~Integrator(){};

    virtual void step_fwd(
        std::vector<BoundPotential *> &bps,
        double *d_x_t,
        double *d_v_t,
        double *d_box_t,
        unsigned int *d_idxs,
        cudaStream_t stream) = 0;

    virtual void initialize(
        std::vector<BoundPotential *> &bps,
        double *d_x_t,
        double *d_v_t,
        double *d_box_t,
        unsigned int *d_idxs,
        cudaStream_t stream) = 0;

    virtual void finalize(
        std::vector<BoundPotential *> &bps,
        double *d_x_t,
        double *d_v_t,
        double *d_box_t,
        unsigned int *d_idxs,
        cudaStream_t stream) = 0;
};

} // end namespace timemachine
