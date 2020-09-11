#pragma once

namespace timemachine {


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

    void compute_block_bounds(
        int N,
        int D,
        const double *coords,
        const double *box,
        const int *perm,
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