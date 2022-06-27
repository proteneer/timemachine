#pragma once

#include "potential.hpp"
#include <vector>

namespace timemachine {

template <typename RealType> class ChiralAtomRestraint : public Potential {

private:
    int *d_idxs_;

    const int R_;

public:
    ChiralAtomRestraint(const std::vector<int> &idxs // [R, 4]
    );

    ~ChiralAtomRestraint();

    virtual void execute_device(
        const int N,
        const int P,
        const double *d_x,
        const double *d_p,
        const double *d_box,
        const double lambda,
        unsigned long long *d_du_dx, // buffered
        unsigned long long *d_du_dp,
        unsigned long long *d_du_dl, // buffered
        unsigned long long *d_u,     // buffered
        cudaStream_t stream) override;
};

} // namespace timemachine
