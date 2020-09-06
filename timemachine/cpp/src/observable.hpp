#pragma once


#include "bound_potential.hpp"


namespace timemachine {

class Observable {

public:

    virtual void collect(
        int step,
        double *d_x_t,
        double *d_box_t,
        double lambda
    ) = 0;

};


// even if we save as frequently as *once every 10 steps* it is still
// much more efficient to separate out the collectors

// dU/dtheta observables
class AvgPartialUPartialTheta : public Observable {

private:

    std::vector<double *> d_du_dp_;
    std::vector<BoundPotential *> bps_;

public:

    AvgPartialUPartialTheta(std::vector<BoundPotential *> bps);

    virtual void collect(
        int step,
        double *d_x_t,
        double *d_box_t,
        double lambda
    ) override;

};

}