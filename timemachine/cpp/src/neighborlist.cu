#include <vector>
#include <iostream>

#include "neighborlist.hpp"
#include "k_find_block_bounds.cuh"
#include "gpu_utils.cuh"

namespace timemachine {

Neighborlist::Neighborlist(
	int N,
	int D) {

    int tpb = 32;
    int B = (N+tpb-1)/tpb;

    gpuErrchk(cudaMalloc(&d_block_bounds_ctr_, B*D*sizeof(*d_block_bounds_ctr_)));
    gpuErrchk(cudaMalloc(&d_block_bounds_ext_, B*D*sizeof(*d_block_bounds_ext_)));

}

Neighborlist::~Neighborlist() {

    gpuErrchk(cudaFree(d_block_bounds_ctr_));
    gpuErrchk(cudaFree(d_block_bounds_ext_));

}


bool is_pow_2(int x) {
    return (x & (x - 1)) == 0;
}

int log2_int(int v) {
    int bits = 0;
    while (v >>= 1) ++bits;
    return bits;
}

int pow_int(int x, int p) {
  if (p == 0) return 1;
  if (p == 1) return x;

  int tmp = pow_int(x, p/2);
  if (p%2 == 0) return tmp * tmp;
  else return x * tmp * tmp;
}


void Neighborlist::compute_block_bounds_cpu(
    const int N,
    const int D,
    const int block_size,
    const double *h_coords,
    const double *h_box,
    double *bb_ctrs,
    double *bb_exts) {
   

    int num_blocks = (N + block_size - 1)/block_size;

    for(int block_idx = 0; block_idx < num_blocks; block_idx++) {
        for(int d=0; d < D; d++) {
            double width = h_box[d*3+d];
            double ci_min =  9999999;
            double ci_max = -9999999;
            for(int i=0; i < block_size; i++) {
                int tid = block_idx*block_size + i;
                if(tid < N) {
                    // int atom_idx = perm[tid];
                    int atom_idx = tid;
                    double ci = h_coords[atom_idx*D + d];
                    ci -= width*floor(ci/width); // move to home box
                    ci_min = ci < ci_min ? ci : ci_min;
                    ci_max = ci > ci_max ? ci : ci_max;
                }
            }
         
            // printf("dim %d block_idx %d ctr %f ext %f\n", d, block_idx, (ci_max + ci_min)/2.0, ci_max - ci_min);
            bb_ctrs[block_idx*D+d] = (ci_max + ci_min)/2.0;
            bb_exts[block_idx*D+d] = (ci_max - ci_min)/2.0;
        }
    }

}

void Neighborlist::build_nblist_cpu(
    int N,
    int D,
    const double *h_coords,
    const double *h_box) {
    /*
    Reference neighborlist algorithm:
    Given N atoms, we subdivide into blocks of size 2^B log_2(N) times.


    Suppose we have 256 atoms. We compute block bounds in groups of
    128, 64, 32, 16, 8, 4, 2, 1. In the limit as the group size == 1 we have the
    coordinates themselves.

    We proceed by

    128x128
    64x64
    32x32
    32x16
    32x8
    32x4
    32x2
    32x1

    When we're finally down to 32x1 we do a full check
   
    */

    if(!is_pow_2(N)) {
        throw std::runtime_error("N is not a power of 2.");
    }

    int row_block_size = N;
    int col_block_size = N;

    int num_rounds = log2_int(N);

    for(int round=0; round < num_rounds; round++)  {
        int block_size = pow_int(2, round);
        std::vector<double> bb_ctrs((N/block_size)*3);
        std::vector<double> bb_exts((N/block_size)*3);
        this->compute_block_bounds_cpu(
            N,
            D,
            block_size,
            h_coords,
            h_box,
            &bb_ctrs[0],
            &bb_exts[0]
        );

    }

    // for(int row=0; row < num_rows; row++) {
    //     for(int col=0; col < num_cols; col++) {

    //     }
    // }
    // 
}

void Neighborlist::compute_block_bounds(
	int N,
	int D,
	const double *coords,
    const double *box,
    const int *perm,
	cudaStream_t stream) {

    int tpb = 32;
    int B = (N+tpb-1)/tpb; // total number of blocks we need to process

    gpuErrchk(cudaMemsetAsync(d_block_bounds_ctr_, 0, B*D*sizeof(*d_block_bounds_ctr_), stream));
    gpuErrchk(cudaMemsetAsync(d_block_bounds_ext_, 0, B*D*sizeof(*d_block_bounds_ext_), stream));

    k_find_block_bounds<<<B, tpb, 0, stream>>>(
        N,
        D,
        B,
        coords,
        box,
        perm,
        d_block_bounds_ctr_,
        d_block_bounds_ext_
    );	

    gpuErrchk(cudaPeekAtLastError());

};


}