#pragma once

#include "potential.hpp"
#include <vector>

namespace timemachine {

template <typename RealType> class ChiralBondRestraint : public Potential {

private:
    const int R_;

    int *d_idxs_;
    int *d_signs_;
    __int128 *d_u_buffer_;

    size_t sum_storage_bytes_;
    void *d_sum_temp_storage_;

public:
    ChiralBondRestraint(
        const std::vector<int> &idxs, // [R, 4]
        const std::vector<int> &signs // [R]
    );

    ~ChiralBondRestraint();

    virtual void execute_device(
        const int N,
        const int P,
        const double *d_x,
        const double *d_p,
        const double *d_box,
        unsigned long long *d_du_dx, // buffered
        unsigned long long *d_du_dp,
        __int128 *d_u,               // buffered
        cudaStream_t stream) override;
};

} // namespace timemachine
