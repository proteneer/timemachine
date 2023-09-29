#pragma once

#include "potential.hpp"
#include <vector>

namespace timemachine {

template <typename RealType> class HarmonicAngle : public Potential {

private:
    int *d_angle_idxs_;
    EnergyType *d_u_buffer_;

    const int A_;

public:
    HarmonicAngle(const std::vector<int> &angle_idxs);

    ~HarmonicAngle();

    virtual void execute_device(
        const int N,
        const int P,
        const double *d_x,
        const ParamsType *d_p,

        const double *d_box,
        unsigned long long *d_du_dx,
        unsigned long long *d_du_dp,
        EnergyType *d_u,
        cudaStream_t stream) override;
};

} // namespace timemachine
