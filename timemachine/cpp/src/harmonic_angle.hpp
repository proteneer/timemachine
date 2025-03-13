#pragma once

#include "potential.hpp"
#include <vector>

namespace timemachine {

template <typename RealType> class HarmonicAngle : public Potential {

private:
    int *d_angle_idxs_;
    __int128 *d_u_buffer_;

    const int A_;

    size_t sum_storage_bytes_;
    void *d_sum_temp_storage_;

public:
    HarmonicAngle(const std::vector<int> &angle_idxs);

    ~HarmonicAngle();

    virtual void execute_device(
        const int N,
        const int P,
        const double *d_x,
        const double *d_p,
        const double *d_box,
        unsigned long long *d_du_dx,
        unsigned long long *d_du_dp,
        __int128 *d_u,
        cudaStream_t stream) override;

    virtual void reset() override {};
};

} // namespace timemachine
