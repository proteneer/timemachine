#pragma once

#include "neighborlist.hpp"
#include "potential.hpp"
#include <vector>

namespace timemachine {

template<typename RealType>
class Nonbonded : public Potential {

private:

    int *d_exclusion_idxs_; // [E,2]
    double *d_scales_; // [E, 2]
    int *d_lambda_plane_idxs_;
    int *d_lambda_offset_idxs_;
    int *p_ixn_count_; // pinned memory

    double beta_;
    double cutoff_;
    Neighborlist<RealType> nblist_;

    const int E_;
    const int N_;

    // reduction buffer
    unsigned long long *d_sorted_du_dl_buffer_;
    unsigned long long *d_sorted_u_buffer_;

    unsigned long long *d_du_dl_buffer_;
    unsigned long long *d_u_buffer_;

    unsigned long long *d_du_dl_reduce_sum_;
    unsigned long long *d_u_reduce_sum_;

    unsigned int *d_perm_; // hilbert curve permutation

    int *d_sorted_lambda_plane_idxs_;
    int *d_sorted_lambda_offset_idxs_;
    double *d_sorted_x_; //
    double *d_sorted_p_; //
    unsigned long long *d_sorted_du_dx_; //
    unsigned long long *d_sorted_du_dp_; //
    unsigned long long *d_du_dp_buffer_; //

    unsigned int *d_bin_to_idx_;
    unsigned int *d_sort_keys_in_;
    unsigned int *d_sort_keys_out_;
    unsigned int *d_sort_vals_in_;
    unsigned int *d_sort_storage_;
    size_t d_sort_storage_bytes_;

    void hilbert_sort(
        const double *d_x,
        const double *d_box,
        cudaStream_t stream
    );

public:

    Nonbonded(
        const std::vector<int> &exclusion_idxs, // [E,2]
        const std::vector<double> &scales, // [E, 2]
        const std::vector<int> &lambda_plane_idxs, // N
        const std::vector<int> &lambda_offset_idxs, // N
        double beta,
        double cutoff
    );

    ~Nonbonded();

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
        cudaStream_t stream
    ) override;



};


}
