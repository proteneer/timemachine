#pragma once

#include "nonbonded_dense.hpp"
#include "nonbonded_pairs.hpp"
#include "potential.hpp"
#include <vector>

namespace timemachine {

template <typename RealType, bool Interpolated> class Nonbonded : public Potential {

private:
    NonbondedDense<RealType, Interpolated> dense_;
    NonbondedPairs<RealType, true, Interpolated> exclusions_; // implement exclusions as negated NonbondedPairs

public:
    // these are marked public but really only intended for testing.
    void set_nblist_padding(double val);
    void disable_hilbert_sort();

    Nonbonded(
        const std::vector<int> &exclusion_idxs,     // [E, 2]
        const std::vector<double> &scales,          // [E, 2]
        const std::vector<int> &lambda_plane_idxs,  // N
        const std::vector<int> &lambda_offset_idxs, // N
        const double beta,
        const double cutoff,
        const std::string &kernel_src);

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

    void du_dp_fixed_to_float(const int N, const int P, const unsigned long long *du_dp, double *du_dp_float) override;
};

} // namespace timemachine
