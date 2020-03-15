
#include "neighborlist.hpp"
#include "k_find_block_bounds.cuh"
#include "kernel_utils.cuh"

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

void Neighborlist::compute_block_bounds(
	int N,
	int D,
	const double *coords,
	cudaStream_t stream) {

    int tpb = 32;
    int B = (N+tpb-1)/tpb;

    gpuErrchk(cudaMemsetAsync(d_block_bounds_ctr_, 0, B*D*sizeof(*d_block_bounds_ctr_), stream));
    gpuErrchk(cudaMemsetAsync(d_block_bounds_ext_, 0, B*D*sizeof(*d_block_bounds_ext_), stream));

    k_find_block_bounds<<<1, B, 0, stream>>>(
        N,
        D,
        B,
        coords,
        d_block_bounds_ctr_,
        d_block_bounds_ext_
    );	

    gpuErrchk(cudaPeekAtLastError());

};


}