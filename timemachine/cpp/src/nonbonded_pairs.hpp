#pragma once

#include "potential.hpp"
#include "vendored/jitify.hpp"
#include <vector>

namespace timemachine {

template <typename RealType, bool Interpolated> class NonbondedPairs : public Potential {

private:
    int *d_pair_idxs_; // [M, 2]
    double *d_scales_; // [M, 2]
    int *d_lambda_plane_idxs_;
    int *d_lambda_offset_idxs_;

    double beta_;
    double cutoff_;

    const int M_; // number of pairs
    const int N_; // number of atoms

    double *d_w_;
    double *d_dw_dl_;

    double *d_dp_dl_;

    int *d_perm_;

    unsigned long long *d_du_dp_buffer_;

    jitify::JitCache kernel_cache_;
    jitify::KernelInstantiation compute_w_coords_instance_;
    jitify::KernelInstantiation compute_permute_interpolated_;
    jitify::KernelInstantiation compute_add_ull_to_real_interpolated_;

public:
    NonbondedPairs(
        const std::vector<int> &pair_idxs,          // [M, 2]
        const std::vector<double> &scales,          // [M, 2]
        const std::vector<int> &lambda_plane_idxs,  // [N]
        const std::vector<int> &lambda_offset_idxs, // [N]
        const double beta,
        const double cutoff,
        const std::string &kernel_src);

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
