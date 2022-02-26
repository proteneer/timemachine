#pragma once

#include "device_buffer.hpp"
#include "potential.hpp"
#include "vendored/jitify.hpp"
#include <vector>

namespace timemachine {

template <typename RealType, bool Negated, bool Interpolated> class NonbondedPairList : public Potential {

private:
    const unsigned int M_; // number of pairs
    const unsigned int N_; // number of atoms

    DeviceBuffer<int> d_pair_idxs_; // [M, 2]
    DeviceBuffer<double> d_scales_; // [M, 2]
    DeviceBuffer<int> d_lambda_plane_idxs_;
    DeviceBuffer<int> d_lambda_offset_idxs_;

    double beta_;
    double cutoff_;

    DeviceBuffer<double> d_w_;
    DeviceBuffer<double> d_dw_dl_;

    DeviceBuffer<double> d_p_interp_;
    DeviceBuffer<double> d_dp_dl_;

    std::unique_ptr<DeviceBuffer<unsigned int>> d_perm_;

    DeviceBuffer<unsigned long long> d_du_dp_buffer_;

    jitify::JitCache kernel_cache_;
    jitify::KernelInstantiation compute_w_coords_instance_;
    jitify::KernelInstantiation compute_permute_interpolated_;
    jitify::KernelInstantiation compute_add_du_dp_interpolated_;

public:
    NonbondedPairList(
        const std::vector<int> &pair_idxs,          // [M, 2]
        const std::vector<double> &scales,          // [M, 2]
        const std::vector<int> &lambda_plane_idxs,  // [N]
        const std::vector<int> &lambda_offset_idxs, // [N]
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
