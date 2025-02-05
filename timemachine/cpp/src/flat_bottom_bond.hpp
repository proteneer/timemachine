#pragma once

#include "potential.hpp"
#include <vector>

namespace timemachine {

template <typename RealType> class FlatBottomBond : public Potential {

private:
    int B_; // TBD make this a constant

    int *d_bond_idxs_;
    __int128 *d_u_buffer_;

    size_t sum_storage_bytes_;
    void *d_sum_temp_storage_;

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
        __int128 *d_u,
        cudaStream_t stream) override;
};

} // namespace timemachine
