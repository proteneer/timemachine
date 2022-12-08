#pragma once

#include "math_utils.cuh"
#include <vector>

namespace timemachine {

// enable 64bit stuff later
template <typename RealType> class Neighborlist {

private:
    const int max_size_; // Max number of atoms the buffers allow
    int N_;              // Number of atoms
    int NC_;             // Number of atoms in column, N_ by default
    int NR_;             // Number of atoms in row, N_ by default

    double *d_row_block_bounds_ctr_;
    double *d_row_block_bounds_ext_;
    double *d_column_block_bounds_ctr_;
    double *d_column_block_bounds_ext_;

    unsigned int *d_row_idxs_;
    unsigned int *d_column_idxs_;

    unsigned int *d_ixn_count_;
    int *d_ixn_tiles_;
    unsigned int *d_ixn_atoms_;
    unsigned int *d_trim_atoms_;

public:
    // N - number of atoms
    Neighborlist(const int N);

    ~Neighborlist();

    void set_row_idxs(std::vector<unsigned int> idxs);

    void reset_row_idxs();

    void reset_row_idxs_device(const cudaStream_t stream);

    void resize(const int size);

    void resize_device(const int size, const cudaStream_t stream);

    void set_idxs_device(
        const int NC, const int NR, unsigned int *column_idxs, unsigned int *row_idxs, const cudaStream_t stream);

    std::vector<std::vector<int>>
    get_nblist_host(const int N, const double *h_coords, const double *h_box, const double cutoff);

    void build_nblist_device(
        const int N, const double *d_coords, const double *d_box, const double cutoff, const cudaStream_t stream);

    void compute_block_bounds_host(
        const int N, const double *h_coords, const double *h_box, double *h_bb_ctrs, double *h_bb_exts);

    unsigned int *get_ixn_atoms() { return d_ixn_atoms_; }

    int *get_ixn_tiles() { return d_ixn_tiles_; }

    unsigned int *get_ixn_count() { return d_ixn_count_; }

    unsigned int *get_row_idxs() { return d_row_idxs_; }

    int get_num_row_idxs() { return NR_; }

    // get max number of row blocks
    int num_row_blocks() const;
    // get max number of column blocks
    int num_column_blocks() const;

    // get max number of interactions
    int max_ixn_count() const;

private:
    // Indicates that should only compute the upper triangle of the interactions matrix, otherwise
    // will compute the entire matrix.
    bool compute_upper_triangular() const;

    // The number of column blocks divided by warp size. Each thread handles a block
    int Y() const;

    void compute_block_bounds_device(
        const int N, const int D, const double *d_coords, const double *d_box, cudaStream_t stream);
};

} // namespace timemachine
