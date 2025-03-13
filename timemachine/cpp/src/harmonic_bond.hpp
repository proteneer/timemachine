#pragma once

#include "potential.hpp"
#include <vector>

namespace timemachine {

template <typename RealType> class HarmonicBond : public Potential {

private:
    const int B_;

    int *d_bond_idxs_;
    __int128 *d_u_buffer_;

    size_t sum_storage_bytes_;
    void *d_sum_temp_storage_;

public:
    int num_bonds() const { return B_; }

    HarmonicBond(const std::vector<int> &bond_idxs // [B, 2]
    );

    ~HarmonicBond();

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

    virtual void reset() override {};
};

} // namespace timemachine
