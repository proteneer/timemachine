#pragma once

#include "kernels/k_nonbonded_common.cuh"
#include "neighborlist.hpp"
#include "potential.hpp"
#include <array>
#include <vector>

namespace timemachine {

template <typename RealType> class NonbondedInteractionGroup : public Potential {

private:
    const int N_; // N_ = NC_ + NR_
    int NR_;      // number of row atoms
    int NC_;      // number of column atoms

    std::array<k_nonbonded_fn, 8> kernel_ptrs_;

    unsigned int *d_col_atom_idxs_;
    unsigned int *d_row_atom_idxs_;

    int *p_ixn_count_; // pinned memory

    double beta_;
    double cutoff_;
    Neighborlist<RealType> nblist_;

    double nblist_padding_;
    double *d_nblist_x_;    // coords which were used to compute the nblist
    double *d_nblist_box_;  // box which was used to rebuild the nblist
    int *d_rebuild_nblist_; // whether or not we have to rebuild the nblist
    int *p_rebuild_nblist_; // pinned
    double *p_box_;

    unsigned int *d_perm_; // hilbert curve permutation

    // "sorted" means
    // - if hilbert sorting enabled, atoms are sorted into contiguous
    //   blocks by interaction group, and each block is hilbert-sorted
    //   independently
    // - otherwise, atoms are sorted into contiguous blocks by
    //   interaction group, with arbitrary ordering within each block
    double *d_sorted_x_; // sorted coordinates
    double *d_sorted_p_; // sorted parameters
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

    void set_atom_idxs_device(
        const int NC, const int NR, unsigned int *d_column_idxs, unsigned int *d_row_idxs, const cudaStream_t stream);

    void set_atom_idxs(const std::vector<int> &atom_idxs);

    NonbondedInteractionGroup(
        const int N, const std::vector<int> &row_atom_idxs, const double beta, const double cutoff);

    ~NonbondedInteractionGroup();

    virtual void execute_device(
        const int N,
        const int P,
        const double *d_x,
        const double *d_p,
        const double *d_box,
        unsigned long long *d_du_dx,
        unsigned long long *d_du_dp,
        unsigned long long *d_u,
        cudaStream_t stream) override;

    void du_dp_fixed_to_float(const int N, const int P, const unsigned long long *du_dp, double *du_dp_float) override;
};

} // namespace timemachine
