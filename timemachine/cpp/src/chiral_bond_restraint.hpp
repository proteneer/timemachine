#pragma once

#include "potential.hpp"
#include <vector>

namespace timemachine {

template <typename RealType> class ChiralBondRestraint : public Potential {

private:
    int *d_idxs_;
    int *d_signs_;
    EnergyType *d_u_buffer_;

    const int R_;

public:
    ChiralBondRestraint(
        const std::vector<int> &idxs, // [R, 4]
        const std::vector<int> &signs // [R]
    );

    ~ChiralBondRestraint();

    virtual void execute_device(
        const int N,
        const int P,
        const CoordsType *d_x,
        const ParamsType *d_p,

        const CoordsType *d_box,
        unsigned long long *d_du_dx, // buffered
        unsigned long long *d_du_dp,
        EnergyType *d_u, // buffered
        cudaStream_t stream) override;
};

} // namespace timemachine
