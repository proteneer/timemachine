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

    double nblist_padding_;
    double *d_nblist_x_; // coords which were used to compute the nblist
    double *d_nblist_box_; // box which was used to rebuild the nblist
    int *d_rebuild_nblist_; // whether or not we have to rebuild the nblist
    int *p_rebuild_nblist_; // pinned

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

    bool compute_4d_;
    bool disable_hilbert_;

    void hilbert_sort(
        const double *d_x,
        const double *d_box,
        cudaStream_t stream
    );

public:

    // these are marked public but really only intended for testing.
    void set_nblist_padding(double val);
    void disable_hilbert_sort();

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
