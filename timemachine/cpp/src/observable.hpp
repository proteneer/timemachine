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
    void avg_du_dp(double *buffer) const;

};



class AvgPartialUPartialLambda : public Observable {

private:

    double *d_sum_du_dl_;
    std::vector<BoundPotential *> bps_;
    int count_;
    int freq_;

    unsigned long long *d_du_dx_t_; // du/dx [N,3]
    unsigned long long *d_du_dl_buffer_; // du/dl [N]
    double *d_sum_storage_;
    size_t d_sum_storage_bytes_;

public:

    AvgPartialUPartialLambda(
        std::vector<BoundPotential *> bps,
        int freq
    );

    ~AvgPartialUPartialLambda();

    virtual void observe(
        int step,
        int N,
        double *d_x_t,
        double *d_box_t,
        double lambda
    ) override;

    // copy into buffer and return shape of params object.
    void avg_du_dl(double *buffer) const;

};


class FullPartialUPartialLambda : public Observable {

private:

    double *d_full_du_dl_;
    std::vector<BoundPotential *> bps_;
    int count_; // current counter
    int size_; // buffer size
    int freq_; // how often we write 

public:

    FullPartialUPartialLambda(
        std::vector<BoundPotential *> bps,
        int freq
    );

    ~FullPartialUPartialLambda();

    virtual void observe(
        int step,
        int N,
        double *d_x_t,
        double *d_box_t,
        double lambda
    ) override;

    int count() const {
        return count_;
    }

    // copy into buffer and return shape of params object.
    void full_du_dl(double *buffer) const;

};

}