#pragma once

#include "potential.hpp"
#include <vector>

namespace timemachine {

template<typename RealType>
class CentroidRestraint : public Potential {

private:

    int *d_group_a_idxs_;
    int *d_group_b_idxs_;
    double *d_masses_;

    int N_;
    int N_A_;
    int N_B_;

    double kb_;
    double b0_;

public:

    CentroidRestraint(
        const std::vector<int> &group_a_idxs,
        const std::vector<int> &group_b_idxs,
        const std::vector<double> &masses,
        const double kb,
        const double b0
    );

    ~CentroidRestraint();

    virtual void execute_device(
        const int N,
        const int P,
        const double *d_x,
        const double *d_p,
        const double *d_box,
        const double lambda,
        unsigned long long *d_du_dx,
        double *d_du_dp,
        double *d_du_dl,
        double *d_u,
        cudaStream_t stream
    ) override;

};


}