#pragma once

#include "potential.hpp"
#include <vector>

namespace timemachine {

// Nonbonded Pair List that computes the interaction energies between pairs of atoms.
// The negated version of this potential should be used in conjunction with a NonbondedAllPairs and/or
// NonbondedInteractionGroup as a way to compute the exclusions and cancel them out from the other potentials
// To ensure valid energies and du_dp values, combine the potentials using a FanoutSummedPotential
template <typename RealType, bool Negated> class NonbondedPairList : public Potential {

private:
    const int M_; // number of pairs

    double beta_;
    double cutoff_;

    __int128 *d_u_buffer_; // [M]

    int *d_pair_idxs_;     // [M, 2]
    double *d_scales_;     // [M, 2]

    size_t sum_storage_bytes_;
    void *d_sum_temp_storage_;

public:
    NonbondedPairList(
        const std::vector<int> &pair_idxs, // [M, 2]
        const std::vector<double> &scales, // [M, 2]
        const double beta,
        const double cutoff);

    ~NonbondedPairList();

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

    void du_dp_fixed_to_float(const int N, const int P, const unsigned long long *du_dp, double *du_dp_float) override;
};

} // namespace timemachine
