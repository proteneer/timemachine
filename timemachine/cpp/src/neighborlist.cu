#include <cassert>
#include <vector>
#include <iostream>

#include "neighborlist.hpp"
#include "k_neighborlist.cuh"
#include "gpu_utils.cuh"

namespace timemachine {

template<typename RealType>
Neighborlist<RealType>::Neighborlist(
    int N) : N_(N) {

    const int B = this->B(); //(N+32-1)/32;
    const int Y = this->Y(); //(B+32-1)/32;

    unsigned long long MAX_TILE_BUFFER = B*B;
    unsigned long long MAX_ATOM_BUFFER = B*B*32;

    // interaction buffers
    gpuErrchk(cudaMalloc(&d_ixn_count_, 1*sizeof(*d_ixn_count_)));
    gpuErrchk(cudaMalloc(&d_ixn_tiles_, MAX_TILE_BUFFER*sizeof(*d_ixn_tiles_)));
    gpuErrchk(cudaMalloc(&d_ixn_atoms_, MAX_ATOM_BUFFER*sizeof(*d_ixn_atoms_)));
    gpuErrchk(cudaMalloc(&d_trim_atoms_, B*Y*32*sizeof(*d_trim_atoms_)));


    // bounding box buffers
    gpuErrchk(cudaMalloc(&d_block_bounds_ctr_, B*3*sizeof(*d_block_bounds_ctr_)));
    gpuErrchk(cudaMalloc(&d_block_bounds_ext_, B*3*sizeof(*d_block_bounds_ext_)));

}

template<typename RealType>
Neighborlist<RealType>::~Neighborlist() {

    gpuErrchk(cudaFree(d_ixn_count_));
    gpuErrchk(cudaFree(d_ixn_tiles_));
    gpuErrchk(cudaFree(d_ixn_atoms_));
    gpuErrchk(cudaFree(d_trim_atoms_));

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

template<typename RealType>
void Neighborlist<RealType>::compute_block_bounds_host(
    const int N,
    const int D,
    const int block_size,
    const double *h_coords,
    const double *h_box,
    double *h_bb_ctrs,
    double *h_bb_exts) {

    assert(N == N_);
    assert(D == 3);
    int h_rebuild = 1;

    double *d_coords = gpuErrchkCudaMallocAndCopy(h_coords, N*3*sizeof(double));
    double *d_box = gpuErrchkCudaMallocAndCopy(h_box, 3*3*sizeof(double));
    int *d_rebuild = gpuErrchkCudaMallocAndCopy(&h_rebuild, 1*sizeof(int));
    cudaStream_t stream = static_cast<cudaStream_t>(0);
    this->compute_block_bounds_device(
        N,
        D,
        d_coords,
        d_box,
        d_rebuild,
        stream
    );

    cudaStreamSynchronize(stream);

    gpuErrchk(cudaMemcpy(h_bb_ctrs, d_block_bounds_ctr_, this->B()*3*sizeof(*d_block_bounds_ctr_), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_bb_exts, d_block_bounds_ext_, this->B()*3*sizeof(*d_block_bounds_ext_), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_coords));
    gpuErrchk(cudaFree(d_box));
    gpuErrchk(cudaFree(d_rebuild));

}

template<typename RealType>
std::vector<std::vector<int> > Neighborlist<RealType>::get_nblist_host(
    int N,
    const double *h_coords,
    const double *h_box,
    const double cutoff) {

    // assert(N==N_);

    int h_rebuild = 1;

    double *d_coords = gpuErrchkCudaMallocAndCopy(h_coords, N*3*sizeof(double));
    double *d_box = gpuErrchkCudaMallocAndCopy(h_box, 3*3*sizeof(double));
    int *d_rebuild = gpuErrchkCudaMallocAndCopy(&h_rebuild, 1*sizeof(int));

    cudaStream_t stream = static_cast<cudaStream_t>(0);
    this->build_nblist_device(
        N,
        d_coords,
        d_box,
        cutoff,
        d_rebuild,
        stream
    );

    cudaStreamSynchronize(stream);

    const int B = this->B(); //(N+32-1)/32;

    unsigned long long MAX_TILE_BUFFER = B*B;
    unsigned long long MAX_ATOM_BUFFER = B*B*32;

    unsigned int h_ixn_count;
    std::vector<int> h_ixn_tiles(MAX_TILE_BUFFER);
    std::vector<unsigned int> h_ixn_atoms(MAX_ATOM_BUFFER);
    gpuErrchk(cudaMemcpy(&h_ixn_count, d_ixn_count_, 1*sizeof(*d_ixn_count_), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&h_ixn_tiles[0], d_ixn_tiles_, MAX_TILE_BUFFER*sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&h_ixn_atoms[0], d_ixn_atoms_, MAX_ATOM_BUFFER*sizeof(unsigned int), cudaMemcpyDeviceToHost));

    std::vector<std::vector<int> > ixn_list(B, std::vector<int>());

    for(int i=0; i < h_ixn_count; i++) {
        int tile_idx = h_ixn_tiles[i];
        for(int j=0; j < 32; j++) {
            int atom_j_idx = h_ixn_atoms[i*32+j];
            if(atom_j_idx < N) {
                ixn_list[tile_idx].push_back(atom_j_idx);
            }
        }
    }

    gpuErrchk(cudaFree(d_coords));
    gpuErrchk(cudaFree(d_box));
    gpuErrchk(cudaFree(d_rebuild));

    return ixn_list;

}

template<typename RealType>
void Neighborlist<RealType>::build_nblist_device(
    const int N,
    const double *d_coords,
    const double *d_box,
    const double cutoff,
    int *d_rebuild_nblist,
    cudaStream_t stream) {

    // assert(N == N_);

    const int D = 3;
    this->compute_block_bounds_device(
        N,
        D,
        d_coords,
        d_box,
        d_rebuild_nblist,
        stream
    );

    const int tpb = 32;
    const int B = this->B(); // (N+32-1)/32;
    const int Y = this->Y(); // (B+32-1)/32;

    dim3 dimGrid(B, Y, 1); // block x, y, z dims

    // (ytz): TBD shared memory, stream
    k_find_blocks_with_ixns<RealType><<<dimGrid, tpb, 0, stream>>>(
        N,
        d_block_bounds_ctr_,
        d_block_bounds_ext_,
        d_coords,
        d_box,
        d_ixn_count_,
        d_ixn_tiles_,
        d_ixn_atoms_,
        d_trim_atoms_,
        cutoff,
        d_rebuild_nblist
    );

    gpuErrchk(cudaPeekAtLastError());

    k_compact_trim_atoms<<<B, tpb, 0, stream>>>(
        N,
        Y,
        d_trim_atoms_,
        d_ixn_count_,
        d_ixn_tiles_,
        d_ixn_atoms_,
        d_rebuild_nblist
    );

    gpuErrchk(cudaPeekAtLastError());

}

template <typename RealType>
void Neighborlist<RealType>::compute_block_bounds_device(
    const int N, // Number of atoms
    const int D, // Box dimensions
    const double *d_coords, // [N*3]
    const double *d_box, // [D*3]
    const int * d_rebuild_nblist,
    cudaStream_t stream) {

    assert(N == N_);
    assert(D == 3);

    const int tpb = 32;
    const int B = (N+tpb-1)/tpb; // total number of blocks we need to process

    k_find_block_bounds<RealType><<<B, tpb, 0, stream>>>(
        N,
        D,
        B,
        d_coords,
        d_box,
        d_block_bounds_ctr_,
        d_block_bounds_ext_,
        d_ixn_count_,
        d_rebuild_nblist
    );	

    gpuErrchk(cudaPeekAtLastError());

};


template class Neighborlist<double>;
template class Neighborlist<float>;


}
