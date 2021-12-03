#pragma once

#include "math_utils.cuh"
#include <vector>

namespace timemachine {

// enable 64bit stuff later
template <typename RealType> class Neighborlist {

private:
    const int NC_;
    const int NR_;

    double *d_col_block_bounds_ctr_;
    double *d_col_block_bounds_ext_;

    double *d_row_block_bounds_ctr_;
    double *d_row_block_bounds_ext_;

    unsigned int *d_ixn_count_;
    int *d_ixn_tiles_;
    unsigned int *d_ixn_atoms_;
    unsigned int *d_trim_atoms_;

public:
    // N - number of atoms
    Neighborlist(const int NC, const int NR = 0);

    ~Neighborlist();

    std::vector<std::vector<int>>
    get_nblist_host(const int NC, const double *h_coords, const double *h_box, const double cutoff);

    std::vector<std::vector<int>> get_nblist_host(
        const int NC,
        const int NR,
        const double *h_column_coords,
        const double *h_row_coords,
        const double *h_box,
        const double cutoff);

    void build_nblist_device(
        const int NC, const double *d_coords, const double *d_box, const double cutoff, cudaStream_t stream);

    void build_nblist_device(
        const int NC,
        const int NR,
        const double *d_col_coords,
        const double *d_row_coords,
        const double *d_box,
        const double cutoff,
        cudaStream_t stream);

    void compute_block_bounds_host(
        const int NC,
        const int D,
        const int block_size,
        const double *h_coords,
        const double *h_box,
        double *h_bb_ctrs,
        double *h_bb_exts);

    unsigned int *get_ixn_atoms() { return d_ixn_atoms_; }

    int *get_ixn_tiles() { return d_ixn_tiles_; }

    unsigned int *get_ixn_count() { return d_ixn_count_; }

    // get max number of row blocks
    int B() const { return ceil_divide(this->compute_full_matrix() ? NR_ : NC_, 32); }

private:
    // Indicates that should compute interactions of all rows and columns
    // when a row is provided, otherwise only compute upper triangle of the ixn matrix.
    bool compute_full_matrix() const { return NR_ > 0; }

    int column_blocks() const { return ceil_divide(NC_, 32); }
    // The number of column blocks divided by warp size. Each thread handles a block
    int Y() const { return ceil_divide(this->column_blocks(), 32); }

    void compute_block_bounds_device(
        const int NC,
        const int NR,
        const int D,
        const double *d_col_coords,
        const double *d_row_coords,
        const double *d_box,
        cudaStream_t stream);
};

} // namespace timemachine
