#pragma once

#include "potential.hpp"
#include <vector>

namespace timemachine {

template <typename RealType> class LogFlatBottomBond : public Potential {

private:
    int *d_bond_idxs_;
    __int128 *d_u_buffer_;

    int B_;
    double beta_;

public:
    int num_bonds() const { return B_; }

    LogFlatBottomBond(const std::vector<int> &bond_idxs, double beta); // [B, 2]

    void set_bonds_device(const int num_bonds, const int *d_bonds, const cudaStream_t stream);

    ~LogFlatBottomBond();

    virtual void execute_device(
        const int N,
        const int P,
        const double *d_x,
        const ParamsType *d_p,

        const double *d_box,
        unsigned long long *d_du_dx,
        unsigned long long *d_du_dp,
        __int128 *d_u,
        cudaStream_t stream) override;
};

} // namespace timemachine
