#include <cassert>
#include <iostream>
#include <vector>

#include "gpu_utils.cuh"
#include "k_neighborlist.cuh"
#include "neighborlist.hpp"

namespace timemachine {

template <typename RealType> Neighborlist<RealType>::Neighborlist(const int NC, const int NR) : NC_(NC), NR_(NR) {

    // NR must be less than NC
    if (NR > NC) {
        throw std::runtime_error("NR is greater than NC");
    }
    const int tpb = warp_size;
    const int column_blocks = this->column_blocks();
    const int row_blocks = this->B();
    const int Y = this->Y();

    unsigned long long MAX_TILE_BUFFER = row_blocks * column_blocks;
    unsigned long long MAX_ATOM_BUFFER = MAX_TILE_BUFFER * tpb;

    // interaction buffers
    gpuErrchk(cudaMalloc(&d_ixn_count_, 1 * sizeof(*d_ixn_count_)));
    gpuErrchk(cudaMalloc(&d_ixn_tiles_, MAX_TILE_BUFFER * sizeof(*d_ixn_tiles_)));
    gpuErrchk(cudaMalloc(&d_ixn_atoms_, MAX_ATOM_BUFFER * sizeof(*d_ixn_atoms_)));
    gpuErrchk(cudaMalloc(&d_trim_atoms_, column_blocks * Y * tpb * sizeof(*d_trim_atoms_)));

    // bounding box buffers
    gpuErrchk(cudaMalloc(&d_col_block_bounds_ctr_, column_blocks * 3 * sizeof(*d_col_block_bounds_ctr_)));
    gpuErrchk(cudaMalloc(&d_col_block_bounds_ext_, column_blocks * 3 * sizeof(*d_col_block_bounds_ext_)));
    if (this->compute_full_matrix()) {
        gpuErrchk(cudaMalloc(&d_row_block_bounds_ctr_, row_blocks * 3 * sizeof(*d_row_block_bounds_ctr_)));
        gpuErrchk(cudaMalloc(&d_row_block_bounds_ext_, row_blocks * 3 * sizeof(*d_row_block_bounds_ext_)));
    } else {
        // If we are using the column as the row, use the block bounds ptr as the row as well
        d_row_block_bounds_ctr_ = d_col_block_bounds_ctr_;
        d_row_block_bounds_ext_ = d_col_block_bounds_ext_;
    }
}

template <typename RealType> Neighborlist<RealType>::~Neighborlist() {

    gpuErrchk(cudaFree(d_ixn_count_));
    gpuErrchk(cudaFree(d_ixn_tiles_));
    gpuErrchk(cudaFree(d_ixn_atoms_));
    gpuErrchk(cudaFree(d_trim_atoms_));

    gpuErrchk(cudaFree(d_col_block_bounds_ctr_));
    gpuErrchk(cudaFree(d_col_block_bounds_ext_));
    if (this->compute_full_matrix()) {
        gpuErrchk(cudaFree(d_row_block_bounds_ctr_));
        gpuErrchk(cudaFree(d_row_block_bounds_ext_));
    }
}

template <typename RealType>
void Neighborlist<RealType>::compute_block_bounds_host(
    const int NC,
    const int D,
    const int block_size,
    const double *h_coords,
    const double *h_box,
    double *h_bb_ctrs,
    double *h_bb_exts) {

    double *d_coords = gpuErrchkCudaMallocAndCopy(h_coords, NC * 3);
    double *d_box = gpuErrchkCudaMallocAndCopy(h_box, 3 * 3);

    this->compute_block_bounds_device(NC, 0, D, d_coords, nullptr, d_box, static_cast<cudaStream_t>(0));
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(
        h_bb_ctrs, d_col_block_bounds_ctr_, this->B() * 3 * sizeof(*d_col_block_bounds_ctr_), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(
        h_bb_exts, d_col_block_bounds_ext_, this->B() * 3 * sizeof(*d_col_block_bounds_ext_), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_coords));
    gpuErrchk(cudaFree(d_box));
}

template <typename RealType>
std::vector<std::vector<int>> Neighborlist<RealType>::get_nblist_host(
    const int NC,
    const int NR,
    const double *h_column_coords,
    const double *h_row_coords,
    const double *h_box,
    const double cutoff) {

    if (NC != NC_) {
        throw std::runtime_error("NC != NC_");
    }
    if (NR != NR_) {
        throw std::runtime_error("NR != NR_");
    }
    if (NR == 0 && h_row_coords != nullptr) {
        throw std::runtime_error("NR == 0, but row coords provided");
    } else if (h_row_coords == nullptr && NR != 0) {
        throw std::runtime_error("No row coords provided, but NR != 0");
    }
    double *d_col_coords = gpuErrchkCudaMallocAndCopy(h_column_coords, NC * 3);

    double *d_row_coords =
        this->compute_full_matrix() ? gpuErrchkCudaMallocAndCopy(h_row_coords, NR * 3) : nullptr;

    double *d_box = gpuErrchkCudaMallocAndCopy(h_box, 3 * 3);

    this->build_nblist_device(NC, NR, d_col_coords, d_row_coords, d_box, cutoff, static_cast<cudaStream_t>(0));

    cudaDeviceSynchronize();
    const int tpb = warp_size;
    const int column_blocks = this->column_blocks();
    const int row_blocks = this->B();
    const int Y = this->Y();

    unsigned long long MAX_TILE_BUFFER = row_blocks * column_blocks;
    unsigned long long MAX_ATOM_BUFFER = MAX_TILE_BUFFER * tpb;

    unsigned int h_ixn_count;
    gpuErrchk(cudaMemcpy(&h_ixn_count, d_ixn_count_, 1 * sizeof(*d_ixn_count_), cudaMemcpyDeviceToHost));
    std::vector<int> h_ixn_tiles(MAX_TILE_BUFFER);
    std::vector<unsigned int> h_ixn_atoms(MAX_ATOM_BUFFER);
    gpuErrchk(cudaMemcpy(&h_ixn_tiles[0], d_ixn_tiles_, MAX_TILE_BUFFER * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(
        cudaMemcpy(&h_ixn_atoms[0], d_ixn_atoms_, MAX_ATOM_BUFFER * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    std::vector<std::vector<int>> ixn_list(row_blocks, std::vector<int>());
    for (int i = 0; i < h_ixn_count; i++) {
        int tile_idx = h_ixn_tiles[i];
        for (int j = 0; j < tpb; j++) {
            int atom_j_idx = h_ixn_atoms[i * tpb + j];
            if (atom_j_idx < NC) {
                ixn_list[tile_idx].push_back(atom_j_idx);
            }
        }
    }
    gpuErrchk(cudaFree(d_col_coords));
    if (this->compute_full_matrix()) {
        gpuErrchk(cudaFree(d_row_coords));
    }
    gpuErrchk(cudaFree(d_box));

    return ixn_list;
}

template <typename RealType>
std::vector<std::vector<int>>
Neighborlist<RealType>::get_nblist_host(int NC, const double *h_coords, const double *h_box, const double cutoff) {
    // Call into the impl that takes column and row coords
    return this->get_nblist_host(NC, 0, h_coords, nullptr, h_box, cutoff);
}

template <typename RealType>
void Neighborlist<RealType>::build_nblist_device(
    const int NC,
    const int NR,
    const double *d_col_coords,
    const double *d_row_coords,
    const double *d_box,
    const double cutoff,
    cudaStream_t stream) {
    gpuErrchk(cudaMemsetAsync(d_ixn_count_, 0, 1 * sizeof(*d_ixn_count_), stream));

    const int D = 3;
    this->compute_block_bounds_device(NC, NR, D, d_col_coords, d_row_coords, d_box, stream);

    const int tpb = warp_size;
    const int column_blocks = this->column_blocks();
    const int row_blocks = this->B();
    const int Y = this->Y();

    dim3 dimGrid(row_blocks, Y, 1); // block x, y, z dims

    // (ytz): TBD shared memory, stream
    if (!this->compute_full_matrix()) {
        // Compute only the upper triangle as rows and cols are the same
        // pass duplicates of column coords and the bounding boxes
        k_find_blocks_with_ixns<RealType, true><<<dimGrid, tpb, 0, stream>>>(
            NC,
            NC,
            d_col_block_bounds_ctr_,
            d_col_block_bounds_ext_,
            d_col_block_bounds_ctr_,
            d_col_block_bounds_ext_,
            d_col_coords,
            d_col_coords,
            d_box,
            d_ixn_count_,
            d_ixn_tiles_,
            d_ixn_atoms_,
            d_trim_atoms_,
            cutoff);
    } else {
        k_find_blocks_with_ixns<RealType, false><<<dimGrid, tpb, 0, stream>>>(
            NC,
            NR,
            d_col_block_bounds_ctr_,
            d_col_block_bounds_ext_,
            d_row_block_bounds_ctr_,
            d_row_block_bounds_ext_,
            d_col_coords,
            d_row_coords,
            d_box,
            d_ixn_count_,
            d_ixn_tiles_,
            d_ixn_atoms_,
            d_trim_atoms_,
            cutoff);
    }

    gpuErrchk(cudaPeekAtLastError());
    k_compact_trim_atoms<<<row_blocks, tpb, 0, stream>>>(
        NC, Y, d_trim_atoms_, d_ixn_count_, d_ixn_tiles_, d_ixn_atoms_);

    gpuErrchk(cudaPeekAtLastError());
}

template <typename RealType>
void Neighborlist<RealType>::build_nblist_device(
    const int NC, const double *d_coords, const double *d_box, const double cutoff, cudaStream_t stream) {

    this->build_nblist_device(NC, 0, d_coords, nullptr, d_box, cutoff, stream);
}

template <typename RealType> int Neighborlist<RealType>::B() const {
    return ceil_divide(this->compute_full_matrix() ? NR_ : NC_, tile_size);
}

template <typename RealType>
void Neighborlist<RealType>::compute_block_bounds_device(
    const int NC,               // Number of atoms in column
    const int NR,               // Number of atoms in row
    const int D,                // Box dimensions
    const double *d_col_coords, // [N*3]
    const double *d_row_coords, // [K*3]
    const double *d_box,        // [D*3]
    cudaStream_t stream) {

    if (NC != NC_) {
        throw std::runtime_error("NC != NC_");
    }
    if (NR != NR_) {
        throw std::runtime_error("NR != NR_");
    }
    if (D != 3) {
        throw std::runtime_error("D != 3");
    }
    if (NR == 0 && d_row_coords != nullptr) {
        throw std::runtime_error("NR == 0, but row coords provided");
    } else if (d_row_coords == nullptr && NR != 0) {
        throw std::runtime_error("No row coords provided, but NR != 0");
    }

    const bool compute_row_bounds = this->compute_full_matrix();

    const int tpb = warp_size;
    const int column_blocks = this->column_blocks(); // total number of blocks we need to process

    k_find_block_bounds<RealType><<<column_blocks, tpb, 0, stream>>>(
        NC, D, column_blocks, d_col_coords, d_box, d_col_block_bounds_ctr_, d_col_block_bounds_ext_);
    gpuErrchk(cudaPeekAtLastError());

    if (this->compute_full_matrix()) {
        const int row_blocks = this->B();
        k_find_block_bounds<RealType><<<row_blocks, tpb, 0, stream>>>(
            NR, D, row_blocks, d_row_coords, d_box, d_row_block_bounds_ctr_, d_row_block_bounds_ext_);
        gpuErrchk(cudaPeekAtLastError());
    }
};

template <typename RealType> bool Neighborlist<RealType>::compute_full_matrix() const { return NR_ > 0; };

template <typename RealType> int Neighborlist<RealType>::column_blocks() const { return ceil_divide(NC_, tile_size); };

template <typename RealType> int Neighborlist<RealType>::Y() const {
    return ceil_divide(this->column_blocks(), warp_size);
};

template class Neighborlist<double>;
template class Neighborlist<float>;

} // namespace timemachine
