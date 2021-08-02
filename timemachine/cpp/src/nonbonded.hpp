#pragma once

#include "jitify.hpp"
#include "neighborlist.hpp"
#include "potential.hpp"
#include <vector>
#include <array>

namespace timemachine {

typedef void (*k_nonbonded_fn)(
    const int N,
    const unsigned int * ixn_count,
    const int * __restrict__ tile_idxs,
    const int * __restrict__ ixn_tiles,
    const unsigned int * __restrict__ ixn_atoms,
    const double * __restrict__ coords,
    const double * __restrict__ params, // [N]
    const double * __restrict__ box,
    const double * __restrict__ dl_dp,
    const double * __restrict__ coords_w, // 4D coords
    const double * __restrict__ dw_dl, // 4D derivatives
    const double lambda,
    const double beta,
    const double cutoff,
    const char * __restrict__ tile_mask,
    unsigned long long * __restrict__ alchemical_du_dx,
    unsigned long long * __restrict__ vanilla_du_dx,
    unsigned long long * __restrict__ alchemical_du_dp,
    unsigned long long * __restrict__ vanilla_du_dp,
    unsigned long long * __restrict__ alchemical_du_dl_buffer,
    unsigned long long * __restrict__ vanilla_du_dl_buffer,
    unsigned long long * __restrict__ alchemical_u_buffer,
    unsigned long long * __restrict__ vanilla_u_buffer,
    const int * __restrict__ run_vanilla
);

template<typename RealType, bool Interpolated>
class Nonbonded : public Potential {

private:

    std::array<k_nonbonded_fn, 16> kernel_ptrs_;

    int *d_exclusion_idxs_; // [E,2]
    double *d_scales_; // [E, 2]
    int *d_lambda_plane_idxs_;
    int *d_lambda_offset_idxs_;
    int *d_lambda_offset_idxs_sorted_; // Needed to determine which tiles are alchemical

    double beta_;
    double cutoff_;
    unsigned int cur_step_;
    Neighborlist<RealType> nblist_;

    const int E_;
    const int N_;

    double nblist_padding_;
    double *d_nblist_x_; // coords which were used to compute the nblist
    double *d_nblist_box_; // box which was used to rebuild the nblist
    int *d_rebuild_nblist_; // whether or not we have to rebuild the nblist

    unsigned int *d_perm_; // hilbert curve permutation

    double *d_w_; //
    double *d_dw_dl_; //

    double *d_sorted_x_; //
    double *d_sorted_w_; //
    double *d_sorted_dw_dl_; //
    double *d_sorted_p_; //
    double *d_unsorted_p_; //
    double *d_sorted_dp_dl_;
    double *d_unsorted_dp_dl_;

    double *d_x_last_; // Unsorted d_x from last run
    double *d_p_last_; // Sorted d_p from last run

    unsigned long long *d_sorted_du_dx_; //
    unsigned long long *d_sorted_du_dp_; //
    unsigned long long *d_du_dp_buffer_; //

    int *d_run_vanilla_tiles_; // whether or not we have to run vanilla tiles

    unsigned long long *d_alchemical_u_; // Energies for Alchemical interactions
    unsigned long long *d_vanilla_u_; // Energies for chemical interactions (not lambda dependent energies)

    unsigned long long *d_alchemical_du_dx_;
    unsigned long long *d_vanilla_du_dx_;

    unsigned long long *d_alchemical_du_dp_;
    unsigned long long *d_vanilla_du_dp_;

    unsigned long long *d_alchemical_du_dl_;
    unsigned long long *d_vanilla_du_dl_;

    char *d_tile_mask_; // Mask that has 1s for alchemical tiles
    char *d_default_tile_mask_; // Mask that has 1s for all tiles
    char *d_compaction_mask_; // Mask used for compaction, either d_tile_mask or d_default_tile_mask values
    int *d_nblist_tiles_; // Tile idxs to be computed after compaction
    int *d_tile_idxs_; // Tile idxs to compact
    int last_kernel_; // Last kernel that was run
    unsigned int *d_compacted_ixn_count_;

    unsigned int *d_partition_storage_;
    size_t d_partition_storage_bytes_;

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
