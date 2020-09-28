#pragma once

#include "neighborlist.hpp"
#include "potential.hpp"
#include <vector>

namespace timemachine {

template<typename RealType>
class Electrostatics : public Potential {

private:

    int *d_exclusion_idxs_; // [E,2]
    double *d_charge_scales_; // [E]

    int *d_lambda_offset_idxs_;

    int *d_perm_;

    double beta_;
    double cutoff_;
    Neighborlist<RealType> nblist_;

    const int E_;
    const int N_;


public:

    Electrostatics(
        const std::vector<int> &exclusion_idxs, // [E,2]
        const std::vector<double> &charge_scales, // [E]
        const std::vector<int> &lambda_offset_idxs, // N
        double beta, // decay in erfc
        double cutoff
    );

    ~Electrostatics();

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