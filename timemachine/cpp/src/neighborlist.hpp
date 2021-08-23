#pragma once

#include <vector>

namespace timemachine {

// enable 64bit stuff later
template<typename RealType>
class Neighborlist {

private:

    const int N_;

    double *d_block_bounds_ctr_;
    double *d_block_bounds_ext_;

    unsigned int *d_ixn_count_;
    int *d_ixn_tiles_;
    unsigned int *d_ixn_atoms_;
    unsigned int *d_trim_atoms_;

public:

    // N - number of atoms
    Neighborlist(
        int N
    );

    ~Neighborlist();

    std::vector<std::vector<int> > get_nblist_host(
        int N,
        const double *h_coords,
        const double *h_box,
        const double cutoff
    );

    void build_nblist_device(
        const int N,
        const double *d_coords,
        const double *d_box,
        const double cutoff,
        cudaStream_t stream
    );

    void compute_block_bounds_host(
        const int N,
        const int D,
        const int block_size,
        const double *h_coords,
        const double *h_box,
        double *h_bb_ctrs,
        double *h_bb_exts
    );

    unsigned int* get_ixn_atoms() {
        return d_ixn_atoms_;
    }

    int* get_ixn_tiles() {
        return d_ixn_tiles_;
    }

    unsigned int* get_ixn_count() {
        return d_ixn_count_;
    }

    // get max number of blocks
    int B() const {
        return (N_+32-1)/32;
    }

private:

    int Y() const {
        return (this->B()+32-1)/32;
    }

    void compute_block_bounds_device(
        int N,
        int D,
        const double *coords,
        const double *box,
        cudaStream_t stream
    );


};

}
