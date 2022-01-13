#pragma once

#include "potential.hpp"
#include <vector>

namespace timemachine {

template <typename RealType> class HarmonicAngle : public Potential {

private:
    int *d_angle_idxs_;
    int *d_lambda_mult_;
    int *d_lambda_offset_;

    const int A_;

public:
    HarmonicAngle(
        const std::vector<int> &angle_idxs, const std::vector<int> &lambda_mult, const std::vector<int> &lambda_offset);

    ~HarmonicAngle();

    virtual void execute_device(
        const int N,
        const int P,
        const double *d_x,
        const double *d_p,
        const double *d_box,
        const double lambda,
        unsigned long long *d_du_dx,
        unsigned long long *d_du_dp,
        unsigned long long *d_du_dl,
        unsigned long long *d_u,
        cudaStream_t stream) override;
};

} // namespace timemachine
