#pragma once

#include <vector>

namespace timemachine {

// enable 64bit stuff later
template<typename RealType>
class Neighborlist {

private:

    int N_;

    double *d_block_bounds_ctr_;
    double *d_block_bounds_ext_;

    unsigned int *d_ixn_count_;
    int *d_ixn_tiles_;
    unsigned int *d_ixn_atoms_;
    unsigned int *d_trim_atoms_;

public:

    Neighborlist(
        int N
    );

    ~Neighborlist();

    // tbd get_hilbert_coords(), get coordinates that are hilbert ordered
    // (note that nonbonded terms also need to sort using the same permutation)

    // non-periodic neighborlist
    std::vector<std::vector<int> > get_nblist_cpu(
        int N,
        const double *h_coords,
        const double *h_box,
        const double cutoff
    );

    void build_nblist_gpu(
        int N,
        const double *d_coords,
        const double *d_box,
        const double cutoff,
        cudaStream_t stream
    );

    // std::vector<std::vector<int> > build_nblist_mpu(
    //     int N,
    //     int D,
    //     const double *h_coords,
    //     const double *h_box,
    //     const double cutoff);

    // void compute_block_bounds_cpu(
    // const int N,
    // const int D,
    // const int block_size,
    // const double *h_coords,
    // const double *h_box,
    // double *bb_ctrs,
    // double *bb_exts);


private:

    int B() const {
        return (N_+32-1)/32;
    }

    int Y() const {
        return (this->B()+32-1)/32;
    }

    void compute_block_bounds(
        int N,
        int D,
        const double *coords,
        const double *box,
        // const int *perm,
        cudaStream_t stream
    );

    double* get_block_bounds_ctr() {
        return d_block_bounds_ctr_;
    }

    double* get_block_bounds_ext() {
        return d_block_bounds_ext_;
    }

};


}