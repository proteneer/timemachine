#pragma once

#include "jitify.hpp"
#include "neighborlist.hpp"
#include "potential.hpp"
#include <vector>
#include <array>

namespace timemachine {

typedef void (*k_nonbonded_fn)(
    const int N,
    const int K,
    const double * __restrict__ coords,
    const double * __restrict__ params, // [N]
    const double * __restrict__ box,
    const double * __restrict__ dl_dp,
    const double * __restrict__ coords_w, // 4D coords
    const double * __restrict__ dw_dl, // 4D derivatives
    const double lambda,
    const double beta,
    const double cutoff,
    const double * __restrict__ shrink_centroid,
    const int * __restrict__ shrink_flags,
    const int * __restrict__ ixn_tiles,
    const unsigned int * __restrict__ ixn_atoms,
    unsigned long long * __restrict__ du_dx,
    unsigned long long * __restrict__ du_dp,
    unsigned long long * __restrict__ du_dl_buffer,
    unsigned long long * __restrict__ u_buffer,
    unsigned long long * __restrict__ centroid_grad);

template<typename RealType, bool Interpolated>
class Nonbonded : public Potential {

private:

    std::array<k_nonbonded_fn, 16> kernel_ptrs_;

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
    const int K_;

    double nblist_padding_;
    double *d_nblist_x_; // coords which were used to compute the nblist
    double *d_nblist_box_; // box which was used to rebuild the nblist
    int *d_rebuild_nblist_; // whether or not we have to rebuild the nblist
    int *p_rebuild_nblist_; // pinned

    unsigned int *d_perm_; // hilbert curve permutation

    double *d_w_; //
    double *d_dw_dl_; //

    double *d_shrink_centroid_;
    unsigned long long *d_shrink_centroid_grad_;
    int *d_shrink_idxs_;
    int *d_shrink_flags_;
    int *d_sorted_shrink_flags_;

    double *d_sorted_x_; //
    double *d_sorted_w_; //
    double *d_sorted_dw_dl_; //
    double *d_sorted_p_; //
    double *d_unsorted_p_; //
    double *d_sorted_dp_dl_;
    double *d_unsorted_dp_dl_;
    unsigned long long *d_sorted_du_dx_; //
    unsigned long long *d_sorted_du_dp_; //
    unsigned long long *d_du_dp_buffer_; //

    unsigned int *d_bin_to_idx_;
    unsigned int *d_sort_keys_in_;
    unsigned int *d_sort_keys_out_;
    unsigned int *d_sort_vals_in_;
    unsigned int *d_sort_storage_;
    size_t d_sort_storage_bytes_;

    bool disable_hilbert_;

    void hilbert_sort(
        const double *d_x,
        const double *d_box,
        cudaStream_t stream
    );

    jitify::JitCache kernel_cache_;
    jitify::KernelInstantiation compute_w_coords_instance_;
    jitify::KernelInstantiation compute_permute_interpolated_;
    jitify::KernelInstantiation compute_add_ull_to_real_interpolated_;

public:

    // these are marked public but really only intended for testing.
    void set_nblist_padding(double val);
    void disable_hilbert_sort();

    Nonbonded(
        const std::vector<int> &exclusion_idxs, // [E,2]
        const std::vector<double> &scales, // [E, 2]
        const std::vector<int> &lambda_plane_idxs, // N
        const std::vector<int> &lambda_offset_idxs, // N
        const double beta,
        const double cutoff,
        const std::vector<int> &shrink_idxs,
        const std::string &kernel_src
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
        unsigned long long *d_du_dl,
        unsigned long long *d_u,
        cudaStream_t stream
    ) override;

};


}
