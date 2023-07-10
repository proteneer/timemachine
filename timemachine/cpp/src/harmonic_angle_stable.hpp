#pragma once

#include "potential.hpp"
#include <vector>

namespace timemachine {

template <typename RealType> class HarmonicAngleStable : public Potential {

private:
    int *d_angle_idxs_;

    const int A_;

public:
    HarmonicAngleStable(const std::vector<int> &angle_idxs);

    ~HarmonicAngleStable();

    virtual void execute_device(
        const int N,
        const int P,
        const double *d_x,
        const double *d_p,
        const double *d_box,
        unsigned long long *d_du_dx,
        unsigned long long *d_du_dp,
        unsigned long long *d_u,
        int *d_u_overflow_count,
        cudaStream_t stream) override;
};

} // namespace timemachine
