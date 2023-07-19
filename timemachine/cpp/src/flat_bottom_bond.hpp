#pragma once

#include "potential.hpp"
#include <vector>

namespace timemachine {

template <typename RealType> class FlatBottomBond : public Potential {

private:
    int *d_bond_idxs_;

    int B_;

public:
    int num_bonds() const { return B_; }

    FlatBottomBond(const std::vector<int> &bond_idxs); // [B, 2]

    void set_bonds_device(const int num_bonds, const int *d_bonds, const cudaStream_t stream);

    ~FlatBottomBond();

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
