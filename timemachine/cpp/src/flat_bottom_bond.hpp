#pragma once

#include "potential.hpp"
#include <vector>

namespace timemachine {

template <typename RealType> class FlatBottomBond : public Potential {

private:
    int *d_bond_idxs_;

    const int B_;

public:
    int num_bonds() const { return B_; }

    FlatBottomBond(const std::vector<int> &bond_idxs); // [B, 2]

    ~FlatBottomBond();

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
