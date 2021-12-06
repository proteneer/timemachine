#pragma once

#include "potential.hpp"
#include <vector>

namespace timemachine {

template <typename RealType> class NonbondedPairs : public Potential {

private:
    int *d_pair_idxs_; // [M, 2]
    double *d_scales_; // [M, 2]

    double beta_;
    double cutoff_;

    const int M_; // number of pairs

    double *d_w_;
    double *d_dw_dl_;

    double *d_p_;
    double *d_dp_dl_;
    unsigned long long *d_du_dp_buffer_;

public:
    NonbondedPairs(
        const std::vector<int> &pair_idxs, // [M, 2]
        const std::vector<double> &scales, // [M, 2]
        const double beta,
        const double cutoff);

    ~NonbondedPairs();

    virtual void execute_device(
        const int N,
        const int P,
        const double *d_x,
        const double *d_p,
        const double *d_box,
        const double lambda,
        unsigned long long *d_du_dx,
        double *d_du_dp,
        unsigned long long *d_du_dl,
        unsigned long long *d_u,
        cudaStream_t stream) override;
};

} // namespace timemachine
