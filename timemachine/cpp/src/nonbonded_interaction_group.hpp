#pragma once

#include "hilbert_sort.hpp"
#include "neighborlist.hpp"
#include "nonbonded_common.hpp"
#include "potential.hpp"
#include <array>
#include <memory>
#include <optional>
#include <vector>

namespace timemachine {

template <typename RealType> class NonbondedInteractionGroup : public Potential {

private:
    const int N_; // total number of atoms, i.e. first dimension of input coords, params
    int NR_;      // number of row atoms
    int NC_;      // number of column atoms

    size_t sum_storage_bytes_;
    void *d_sum_temp_storage_;

    std::array<k_nonbonded_fn, 8> kernel_ptrs_;

    unsigned int *d_col_atom_idxs_;
    unsigned int *d_row_atom_idxs_;

    int *p_ixn_count_; // pinned memory

    double beta_;
    double cutoff_;
    // This is safe to overflow, either reset to 0 or increment
    unsigned int steps_since_last_sort_;
    Neighborlist<RealType> nblist_;

    const double nblist_padding_;
    __int128 *d_u_buffer_;  // [NONBONDED_KERNEL_BLOCKS]
    double *d_nblist_x_;    // coords which were used to compute the nblist
    double *d_nblist_box_;  // box which was used to rebuild the nblist
    int *d_rebuild_nblist_; // whether or not we have to rebuild the nblist
    int *p_rebuild_nblist_; // pinned

    unsigned int *d_perm_;  // hilbert curve permutation

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

    std::unique_ptr<HilbertSort> hilbert_sort_;

    cudaEvent_t nblist_flag_sync_event_; // Event to synchronize rebuild flag on

    const bool disable_hilbert_;

    bool needs_sort();

    void sort(const double *d_x, const double *d_box, cudaStream_t stream);

    void validate_idxs(
        const int N,
        const std::vector<int> &row_atom_idxs,
        const std::vector<int> &col_atom_idxs,
        const bool allow_empty);

public:
    virtual void reset() override;

    void set_atom_idxs_device(
        const int NC, const int NR, unsigned int *d_column_idxs, unsigned int *d_row_idxs, const cudaStream_t stream);

    void set_atom_idxs(const std::vector<int> &row_atom_idxs, const std::vector<int> &col_atom_idxs);

    NonbondedInteractionGroup(
        const int N,
        const std::vector<int> &row_atom_idxs,
        const std::vector<int> &col_atom_idxs,
        const double beta,
        const double cutoff,
        const bool disable_hilbert_sort = false,
        const double nblist_padding = 0.1);

    ~NonbondedInteractionGroup();

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
