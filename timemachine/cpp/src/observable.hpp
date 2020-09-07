#pragma once

#include <vector>
#include "bound_potential.hpp"


namespace timemachine {

class Observable {

public:

    virtual void observe(
        int step,
        int N,
        double *d_x_t,
        double *d_box_t,
        double lambda
    ) = 0;

};


class AvgPartialUPartialParam : public Observable {

private:

    double *d_sum_du_dp_;
    BoundPotential *bp_;
    int count_;
    int freq_;

public:

    AvgPartialUPartialParam(
        BoundPotential *bp,
        int freq
    );

    ~AvgPartialUPartialParam();

    virtual void observe(
        int step,
        int N,
        double *d_x_t,
        double *d_box_t,
        double lambda
    ) override;

    std::vector<int> shape() const {
        return this->bp_->shape;
    }
    // copy into buffer and return shape of params object.
    void avg_du_dp(double *buffer);




};

// TBD
// AvgPartialUPartialLambda


}