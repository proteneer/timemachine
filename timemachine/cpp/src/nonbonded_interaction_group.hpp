#pragma once

#include "neighborlist.hpp"
#include "nonbonded_common.cuh"
#include "potential.hpp"
#include <array>
#include <set>
#include <vector>

namespace timemachine {

template <typename RealType, bool Interpolated> class NonbondedInteractionGroup : public Potential {

private:
    const int N_;  // N_ = NC_ + NR_
    const int NR_; // number of row atoms
    const int NC_; // number of column atoms

    std::array<k_nonbonded_fn, 16> kernel_ptrs_;

    unsigned int *d_col_atom_idxs_;
    unsigned int *d_row_atom_idxs_;

    int *d_lambda_plane_idxs_;
    int *d_lambda_offset_idxs_;
    int *p_ixn_count_; // pinned memory

    double beta_;
    double cutoff_;
    Neighborlist<RealType> nblist_;

    double nblist_padding_;
    double *d_nblist_x_;    // coords which were used to compute the nblist
    double *d_nblist_box_;  // box which was used to rebuild the nblist
    int *d_rebuild_nblist_; // whether or not we have to rebuild the nblist
    int *p_rebuild_nblist_; // pinned

    unsigned int *d_perm_; // hilbert curve permutation

    double *d_w_; // 4D coordinates
    double *d_dw_dl_;

    // "sorted" means
    // - if hilbert sorting enabled, atoms are sorted into contiguous
    //   blocks by interaction group, and each block is hilbert-sorted
    //   independently
    // - otherwise, atoms are sorted into contiguous blocks by
    //   interaction group, with arbitrary ordering within each block
    double *d_sorted_x_; // sorted coordinates
    double *d_sorted_w_; // sorted 4D coordinates
    double *d_sorted_dw_dl_;
    double *d_sorted_p_; // sorted parameters
    double *d_sorted_dp_dl_;
    unsigned long long *d_sorted_du_dx_;
    unsigned long long *d_sorted_du_dp_;
    unsigned long long *d_du_dp_buffer_;

    // used for hilbert sorting
    unsigned int *d_bin_to_idx_; // mapping from 256x256x256 grid to hilbert curve index
    unsigned int *d_sort_keys_in_;
    unsigned int *d_sort_keys_out_;
    unsigned int *d_sort_vals_in_;
    unsigned int *d_sort_storage_;
    size_t d_sort_storage_bytes_;

    bool disable_hilbert_;

    void hilbert_sort(
        const int N,
        const unsigned int *d_atom_idxs,
        const double *d_x,
        const double *d_box,
        unsigned int *d_perm,
        cudaStream_t stream);

public:
    // these are marked public but really only intended for testing.
    void set_nblist_padding(double val);
    void disable_hilbert_sort();

    NonbondedInteractionGroup(
        const std::set<int> &row_atom_idxs,
        const std::vector<int> &lambda_plane_idxs,  // N
        const std::vector<int> &lambda_offset_idxs, // N
        const double beta,
        const double cutoff);

    ~NonbondedInteractionGroup();

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
