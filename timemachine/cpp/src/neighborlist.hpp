#pragma once

#include <vector>

namespace timemachine {


template<typename RealType>
class Neighborlist {

private:

    double *d_block_bounds_ctr_;
    double *d_block_bounds_ext_;

public:

    Neighborlist(
        int N,
        int D
    );

    ~Neighborlist();

    // tbd get_hilbert_coords(), get coordinates that are hilbert ordered
    // (note that nonbonded terms also need to sort using the same permutation)

    // non-periodic neighborlist
    std::vector<std::vector<int> > build_nblist_cpu(
        int N,
        int D,
        const double *h_coords,
        const double *h_box,
        const double cutoff);

    std::vector<std::vector<int> > build_nblist_mpu(
        int N,
        int D,
        const double *h_coords,
        const double *h_box,
        const double cutoff);

    void compute_block_bounds_cpu(
    const int N,
    const int D,
    const int block_size,
    const double *h_coords,
    const double *h_box,
    double *bb_ctrs,
    double *bb_exts);


private:

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