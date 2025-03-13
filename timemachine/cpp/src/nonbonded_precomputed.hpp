#pragma once

#include "potential.hpp"
#include <vector>

namespace timemachine {

template <typename RealType> class NonbondedPairListPrecomputed : public Potential {

private:
    const int B_;

    const double beta_;
    const double cutoff_;

    int *d_idxs_;
    __int128 *d_u_buffer_;

    size_t sum_storage_bytes_;
    void *d_sum_temp_storage_;

public:
    int num_bonds() const { return B_; }

    NonbondedPairListPrecomputed(
        const std::vector<int> &pair_idxs, // [B, 2]
        const double beta,
        const double cutoff);

    ~NonbondedPairListPrecomputed();

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

    void du_dp_fixed_to_float(const int N, const int P, const unsigned long long *du_dp, double *du_dp_float) override;

    virtual void reset() override {};
};

} // namespace timemachine
