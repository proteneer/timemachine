#pragma once

#include "neighborlist.hpp"
#include "nonbonded_common.cuh"
#include "potential.hpp"
#include "vendored/jitify.hpp"
#include <array>
#include <optional>
#include <set>
#include <vector>

namespace timemachine {

template <typename RealType, bool Interpolated> class NonbondedAllPairs : public Potential {

private:
    const int N_; // total number of atoms, i.e. first dimension of input coords, params
    int K_;       // number of interacting atoms, K_ <= N_

    int *d_lambda_plane_idxs_;
    int *d_lambda_offset_idxs_;

    double beta_;
    double cutoff_;

    unsigned int *d_atom_idxs_; // [K_] indices of interacting atoms

    Neighborlist<RealType> nblist_;
    int *p_ixn_count_; // pinned memory

    double nblist_padding_;
    double *d_nblist_x_;    // coords which were used to compute the nblist
    double *d_nblist_box_;  // box which was used to rebuild the nblist
    int *d_rebuild_nblist_; // whether or not we have to rebuild the nblist
    int *p_rebuild_nblist_; // pinned

    double *d_w_; // 4D coordinates
    double *d_dw_dl_;

    // "gathered" arrays represent the subset of atoms specified by
    // atom_idxs (if the latter is specified, otherwise all atoms).
    //
    // If hilbert sorting is enabled, "gathered" arrays are sorted by
    // hilbert curve index; otherwise, the ordering is that specified
    // by atom_idxs (or the input ordering, if atom_idxs is not
    // specified)
    unsigned int *d_sorted_atom_idxs_; // [K_] indices of interacting atoms, sorted by hilbert curve index
    double *d_gathered_x_;             // sorted coordinates for subset of atoms
    double *d_gathered_w_;             // sorted 4D coordinates for subset of atoms
    double *d_gathered_dw_dl_;
    double *d_gathered_p_; // sorted parameters for subset of atoms
    double *d_gathered_dp_dl_;
    unsigned long long *d_gathered_du_dx_;
    unsigned long long *d_gathered_du_dp_;
    unsigned long long *d_du_dp_buffer_;

    // used for hilbert sorting
    unsigned int *d_bin_to_idx_; // mapping from 256x256x256 grid to hilbert curve index
    unsigned int *d_sort_keys_in_;
    unsigned int *d_sort_keys_out_;
    unsigned int *d_sort_vals_in_;
    unsigned int *d_sort_storage_;
    size_t d_sort_storage_bytes_;

    bool disable_hilbert_;

    std::array<k_nonbonded_fn, 16> kernel_ptrs_;

    jitify::JitCache kernel_cache_;
    jitify::KernelInstantiation compute_w_coords_instance_;
    jitify::KernelInstantiation compute_gather_interpolated_;
    jitify::KernelInstantiation compute_add_du_dp_interpolated_;

    void hilbert_sort(const double *d_x, const double *d_box, cudaStream_t stream);
    void verify_atom_idxs(const std::vector<unsigned int> &atom_idxs);

public:
    // these are marked public but really only intended for testing.
    void set_nblist_padding(double val);
    void disable_hilbert_sort();

    NonbondedAllPairs(
        const std::vector<int> &lambda_plane_idxs,  // N
        const std::vector<int> &lambda_offset_idxs, // N
        const double beta,
        const double cutoff,
        const std::optional<std::set<unsigned int>> &atom_idxs,
        const std::string &kernel_src);

    ~NonbondedAllPairs();

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

    void set_atom_idxs(const std::vector<unsigned int> &atom_idxs);
    void set_atom_idxs_device(const int K, const unsigned int *d_atom_idxs, const cudaStream_t stream);

    void du_dp_fixed_to_float(const int N, const int P, const unsigned long long *du_dp, double *du_dp_float) override;
};

} // namespace timemachine
