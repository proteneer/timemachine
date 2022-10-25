#pragma once

#include "potential.hpp"
#include <vector>

namespace timemachine {

template <typename RealType> class NonbondedPairListPrecomputed : public Potential {

private:
    int *d_idxs_;
    const int B_;

    double beta_;
    double cutoff_;

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
        const double lambda,
        unsigned long long *d_du_dx, // buffered
        unsigned long long *d_du_dp,
        unsigned long long *d_du_dl, // buffered
        unsigned long long *d_u,     // buffered
        cudaStream_t stream) override;

    void du_dp_fixed_to_float(const int N, const int P, const unsigned long long *du_dp, double *du_dp_float) override;
};

} // namespace timemachine
