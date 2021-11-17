#pragma once

#include "neighborlist.hpp"
#include "potential.hpp"
#include <vector>

namespace timemachine {

template <typename RealType> class LennardJones : public Potential {

private:
    double *d_lj_params_; // [N, 2]

    int *d_exclusion_idxs_; // [E,2]
    double *d_lj_scales_;   // [E]
    int *d_lambda_plane_idxs_;
    int *d_lambda_offset_idxs_;

    int *d_perm_;

    double cutoff_;
    Neighborlist<RealType> nblist_;

    const int E_;
    const int N_;

public:
    LennardJones(
        const std::vector<int> &exclusion_idxs,     // [E,2]
        const std::vector<double> &lj_scales,       // [E]
        const std::vector<int> &lambda_plane_idxs,  // N
        const std::vector<int> &lambda_offset_idxs, // N
        double cutoff);

    ~LennardJones();

    virtual void execute_device(
        const int N,
        const int P,
        const double *d_x,
        const double *d_p,
        const double *d_box,
        const double lambda,
        unsigned long long *d_du_dx,
        double *d_du_dp,
        double *d_du_dl,
        double *d_u,
        cudaStream_t stream) override;
};

} // namespace timemachine
